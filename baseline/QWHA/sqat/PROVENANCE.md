# QWHA baseline — what is upstream, what is ours, and why

QWHA (*Quantization-Aware Walsh-Hadamard Adaptation*, [arXiv:2509.17428](https://arxiv.org/abs/2509.17428))
reproduced on **Llama-2-7B, Commonsense-170k, 1 epoch**, in this repo's span cell, for the INT3 g64
and INT2 g32 rows of `RESULTS_SUMMARY.md`.

Upstream lives one directory up (`baseline/QWHA/`, github.com/vantaa89/QWHA @ `fc8d288`, vendored
whole, including its `peft` fork and `MagR/`). **Nothing upstream was edited.** Everything in this
`sqat/` directory is the harness around it.

## The method, unchanged

* plain-GPTQ INT-b asymmetric base (`GPTQConfig(sym=False, dataset="wikitext2")` → optimum →
  gptqmodel, 128×2048 calibration sequences, `desc_act` off) — upstream's own GPTQ branch;
* a Walsh-Hadamard adapter at the LoRA-r64 parameter budget, `n_frequency = 64 · (in + out)` per
  layer, on all seven projections;
* AdaAlloc initialization (α = 1) from the quantization error, refined against the wikitext2
  `Xᵀ X` root — `initialize.initialize_adapter`, called as-is;
* adapter scaling 4000 (paper Table 8), spectrum the only trainable tensor.

## What this harness adds, and why

| File | Why it exists |
|---|---|
| `setup_env.sh` | Isolated conda env (`/scratch/.../conda_envs/qwha`): python 3.12, torch 2.5.1+cu124, transformers 4.51.3, gptqmodel 2.2.0, `fast-hadamard-transform`, the vendored peft fork. The training env (`saltq`, torch 2.11) and the eval env (`vllm-eval`) are untouched. |
| `qwha_common.py` | Explicit base path + explicit device (upstream keys the GPTQ cache by model id only — a second group size would overwrite the first — and ends in `.cuda()` / `device_map="auto"`, neither of which survives DDP). Also writes the per-layer indices into the *registered buffer* as well as the python dict; upstream writes only the dict, so a trained checkpoint would ship indices that do not match its spectrum. |
| `quantize_base.py` | Upstream's GPTQ branch, lifted out of the training path so quantization happens once, not per rank. |
| `init_adapter.py` | Upstream's initialization, with the calibration pass and the eigendecompositions split so the fp32 base can leave the 40 GB card in between (upstream ran on 80 GB). Same damping, same clamp, same `R`. |
| `train_commonsense.py` | This repo's cell: `datasets/commonsense`, the repo PROMPT, `loss_span=instruction+response` — tokenization imported from `src/data.py` rather than copied. |
| `export_dense.py` | A QWHA layer is exactly one dense linear map, `W_eff = iWHT(WHT(dequant(W_q)) + S)`; the export materializes it so `runs/eval_vllm.sh` can serve the model like every other row, and **checks** the identity per layer against the live module. |
| `smoke_test.sh` | The whole chain on a 2-layer random Llama, ~10 minutes on 2 GPUs. |

## Choices that are not upstream defaults

* **No MagR.** Upstream's README quantizes with MagR + GPTQ. Every other row of these tables
  (GPTQ floor, QA-LoRA, SALT-Q) sits on a plain GPTQ base, and a baseline on a stronger base
  measures the base rather than the adapter. Plain GPTQ it is; `MagR/` is vendored if that
  comparison is ever wanted.
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
qsub jobs/qwha_smoke.pbs                             # harness check, 2 GPUs, ~10 min
qsub jobs/qwha_prep_int3_g64.pbs                     # 1 GPU: GPTQ base + AdaAlloc init
qsub -W depend=afterok:<prep-id> jobs/qwha_int3_g64_span.pbs   # 4 GPUs: train + export + eval
```

Every stage is idempotent — it skips when its output exists — so a job that hits the 24 h
walltime can simply be resubmitted. Artefacts land in `$QWHA_CACHE_PATH`
(`/scratch/.../SQAT_outputs/qwha_cache`: `gptq_models/`, `initialized_checkpoints/`) and in
`outputs/qwha_cs170k_*` (adapter) / `outputs/qwha_cs170k_*-dense` (what vLLM reads). Scores land
in `results/commonsense_vllm/<tag>.json` like every other row.
