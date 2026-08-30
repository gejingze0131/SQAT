# QWHA baseline — what is upstream, what is ours, and why

QWHA (*Quantization-Aware Walsh-Hadamard Adaptation*, [arXiv:2509.17428](https://arxiv.org/abs/2509.17428))
reproduced on **Llama-2-7B, Commonsense-170k, 1 epoch**, in this repo's span cell, for the INT3 g64
and INT2 g32 rows of `RESULTS_SUMMARY.md`.

Upstream lives one directory up (`baseline/QWHA/`, github.com/vantaa89/QWHA @ `fc8d288`, vendored
whole, including its `peft` fork and `MagR/`). **Nothing upstream was edited.** Everything in this
`sqat/` directory is the harness around it.

## The method, unchanged

* a Walsh-Hadamard adapter at the LoRA-r64 parameter budget, `n_frequency = 64 · (in + out)` per
  layer, on all seven projections;
* AdaAlloc initialization (α = 1) from the quantization error, refined against the calibration
  `Xᵀ X` root — `initialize.initialize_adapter` and `compute_quant_error`, called as-is;
* adapter scaling 4000 (paper Table 8), spectrum the only trainable tensor;
* asymmetric INT-b group-wise base, `desc_act` off.

## The calibration — the one deliberate departure from upstream

Upstream calibrates both the base (optimum `GPTQConfig(dataset="wikitext2")`) and the AdaAlloc
`Xᵀ X` on generic text. This repo measured that as the failure mode, not a detail: the same
merged checkpoint scored **INT2 36.64** on the old first-N calibration (100 % BoolQ, 9.5k tokens)
and **66.22** on a task-balanced 3500-record in-domain set (471k tokens); INT3 went 72.60 → 76.05.
Standard C4 128×2048 loses the instruction template outright (30.67). A QWHA row on a
wikitext2-calibrated base would be reporting the calibration rather than the adapter.

So both calibrations move to the balanced in-domain set every `bcal` row uses
(`src/data.load_calibration_data`, `calibration_samples: 3500`, `calibration_sampling: balanced`,
`calibration_seq_len: 2048`, prompt template included):

* **the base** — `make_bcal_base.py` runs THIS repo's `src/gptq.py` on those records and PACKS the
  resulting grid into GPTQModel's checkpoint format (the format upstream's peft fork wraps).
  Packing, not re-quantizing: `pack()` recovers codes as `round(W/s + z)`, so our dequantized
  weights with our `(scale, zero)` reproduce our codes exactly — asserted by a round trip against
  the GPTQ output before anything trains on it. `qat.gptq.nsamples` must equal
  `qat.sqat.calibration_samples`, because `gptq_quantize_model_sequential` consumes only the first
  `nsamples` records.
* **the AdaAlloc `Xᵀ X`** — `init_adapter.py --calib balanced`, over the same records, with the
  padding masked out of the accumulation (upstream's dense wikitext2 windows had none).

`quantize_base.py` keeps upstream's wikitext2 path for provenance and for a calibration ablation
on an otherwise identical pipeline.

## What this harness adds, and why

| File | Why it exists |
|---|---|
| `setup_env.sh` | Isolated conda env (`/scratch/.../conda_envs/qwha`): python 3.12, torch 2.5.1+cu124, transformers 4.51.3, gptqmodel 2.2.0, `fast-hadamard-transform`, the vendored peft fork. The training env (`saltq`, torch 2.11) and the eval env (`vllm-eval`) are untouched. |
| `qwha_common.py` | Explicit base path + explicit device (upstream keys the GPTQ cache by model id only — a second group size would overwrite the first — and ends in `.cuda()` / `device_map="auto"`, neither of which survives DDP). Also writes the per-layer indices into the *registered buffer* as well as the python dict; upstream writes only the dict, so a trained checkpoint would ship indices that do not match its spectrum. |
| `make_bcal_base.py` | The base the reported rows use: this repo's GPTQ grid on the balanced in-domain calibration, packed into GPTQModel's format, round trip asserted. |
| `quantize_base.py` | Upstream's optimum/wikitext2 GPTQ branch, lifted out of the training path. Kept for provenance; not used by the reported rows. |
| `init_adapter.py` | Upstream's initialization, on the balanced records (`--calib balanced`, padding masked) or upstream's wikitext2 (`--calib wikitext2`), with the calibration pass and the eigendecompositions split so the fp32 base can leave the 40 GB card in between (upstream ran on 80 GB). Same damping, same clamp, same `R`. |
| `prefetch_data.sh` | Login-node prefetch: compute nodes have no route out, and a bare hub snapshot is not enough for `load_dataset` under `HF_DATASETS_OFFLINE`. |
| `train_commonsense.py` | This repo's cell: `datasets/commonsense`, the repo PROMPT, `loss_span=instruction+response` — tokenization imported from `src/data.py` rather than copied. |
| `export_dense.py` | A QWHA layer is exactly one dense linear map, `W_eff = iWHT(WHT(dequant(W_q)) + S)`; the export materializes it so `runs/eval_vllm.sh` can serve the model like every other row, and **checks** the identity per layer against the live module. |
| `smoke_test.sh` | The whole chain on a 2-layer random Llama, ~10 minutes on 2 GPUs. |

## Choices that are not upstream defaults

* **No MagR.** Upstream's README quantizes with MagR + GPTQ. Every other row of these tables
  (GPTQ floor, QA-LoRA, SALT-Q) sits on a plain GPTQ base, and a baseline on a stronger base
  measures the base rather than the adapter. Plain GPTQ it is; `MagR/` is vendored if that
  comparison is ever wanted.
* **Balanced in-domain calibration**, for the base and the AdaAlloc `Xᵀ X` alike — see the section
  above. This is what puts the row in the `bcal` cell, and it is only comparable to `bcal` rows.
* **Learning rate is QWHA's own**, not this repo's. The paper (Table 9) has no Llama-2-7B entry,
  so the Llama-3.1-8B / Alpaca column is used — instruction tuning, nearest model:
  **3e-5 at INT3, 2e-5 at INT2**. A baseline retuned to match the method under test is not a
  baseline.
* **Recipe is this repo's**: 1 epoch, effective batch 80, max_seq_len 2048, cosine, warmup 0.03,
  weight decay 0.01, max_grad_norm 0.3, bf16, `group_by_length: false`, seed 42, gradient
  checkpointing on. (Upstream's Alpaca recipe is 3 epochs, batch 64, seq 512, warmup 0.1,
  wd 1.0, no checkpointing.) The cell is the comparison; the method keeps its own knobs.

## Deployment caveat — the row is "b + fp16"

QA-LoRA and SALT-Q deploy as pure INT-b. QWHA does not: the trained spectrum is ~160M fp16 values
(the r=64 budget) with no lossless merge into b-bit weights, and its forward needs a WHT of the
dequantized weight on top. The dense fp16 export is a numerical stand-in for evaluation only, and
the tables should read the row as **3 + fp16** / **2 + fp16**, next to the `fp16-salient PTQ` rows
rather than next to the pure-INT rows.

## Running it

```bash
bash baseline/QWHA/sqat/setup_env.sh                 # once, login node
bash baseline/QWHA/sqat/prefetch_data.sh             # once, login node
qsub jobs/qwha_smoke.pbs                             # harness check, 2 GPUs, ~10 min
qsub jobs/qwha_prep_int3_g64_bcal.pbs                # 1 GPU: bcal GPTQ base + AdaAlloc init
qsub -W depend=afterok:<prep-id> jobs/cs_qwha_int3_g64_span_bcal.pbs   # 4 GPUs: train+export+eval
```

The pipeline is `runs/qwha/run_qwha_commonsense.sh` → `runs/qwha/_pipeline.sh`, the same shape as
`runs/saltq/_pipeline.sh` and `runs/lota/_pipeline.sh`: base → init → train → export → eval →
`results_saltq.csv` (`--filter qwha`), all into one log under `logs/commonsense_170k/`.

Every stage is idempotent — it skips when its output exists — so a job that hits the 24 h
walltime can simply be resubmitted. Artefacts land in `$QWHA_CACHE_PATH`
(`/scratch/.../SQAT_outputs/qwha_cache`: `gptq_models/`, `initialized_checkpoints/`) and in
`outputs/qwha_cs170k_*` (adapter) / `outputs/qwha_cs170k_*-dense` (what vLLM reads). Scores land
in `results/commonsense_vllm/<tag>.json` like every other row.
