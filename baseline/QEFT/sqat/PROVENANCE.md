# QEFT baseline — what is upstream, what is ours, and why

QEFT (*Quantization for Efficient Fine-Tuning of LLMs*, Lee, Jin, Cho, Park,
[Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.811/),
[arXiv:2410.08661](https://arxiv.org/abs/2410.08661), code
[xvyaward/qeft](https://github.com/xvyaward/qeft)) reproduced on **Llama-2-7B,
Commonsense-170k, 1 epoch**, in this repo's span cell, for the INT3 g64 and INT2 g32 rows of
`RESULTS_SUMMARY.md`.

## The method

Three parts, all of them kept:

1. **Weak columns in FP16** (inherited from OWQ). Per linear layer the `k` input columns whose
   quantization hurts most are left at full precision and everything else is group-wise INT-b.
   The sensitivity is second-order — `E_i ≈ ΔW_i,: H ΔW_i,:ᵀ` with `H = 2 X Xᵀ`, so the per-column
   term is `λ_j = diag(H)_j` (paper Eq. 2–3).
2. **Offline Global Reordering (OGR)**, the paper's own contribution. OWQ's weak columns are
   scattered, which is what forces its irregular kernel. QEFT observes that the weak indices
   largely coincide across layers (activation outliers propagate along the residual stream) and
   takes **one global index set for the whole network**, folding its permutation offline into the
   embedding, both norms of every block, the q/k/v/gate/up input columns, the o_proj/down_proj
   output rows and the lm_head. One segment, so — unlike SALT-Q — there is no runtime gather
   anywhere and the exported checkpoint is a plain Llama.
3. **Weak Column Tuning (WCT)**. Fine-tuning trains the fp16 weak columns and nothing else: no
   adapter, no fake-quant, the quantized 98–99 % frozen with zero degrees of freedom. The weight
   gradient is a `[out, k]` block, so it costs `k/IC` of a dense one and only the weak columns'
   activations have to be kept (paper §3.3).

Selection follows the released code, which differs slightly from Algorithm 1 in the paper:
`extract_outidx.py` accumulates the **whole** normalized `λ` vector of every q/k/v/gate/up layer
(`sensitivity_sum += λ / λ.mean()`) rather than only each layer's local top-k, and then takes the
global top-k. `qeft_common.select_global_weak_columns` implements the released version; the
per-layer mean normalization is what stops one loud layer from deciding the global set on its own
(`scripts/test_qeft.py` checks exactly that).

## What this harness is, and why there is no vendored upstream

Unlike `baseline/LoTA-QAF/` and `baseline/QWHA/`, **no upstream code is vendored and there is no
third conda env**. QEFT's repository is a fork of OWQ built around custom CUDA kernels for its
packed mixed-precision format, pinned to torch 2.0 / an old transformers. None of that is
reachable from this repo's measurement: every row here is scored by generating with vLLM from a
**dense checkpoint**, so the kernels only buy inference speed, which this table does not measure.
What is left of the method — the sensitivity statistic, the global reordering, the GPTQ that skips
the weak columns, and training only those columns — is arithmetic this repo already implements for
its own methods, so QEFT is implemented against it directly, in the `saltq` env:

| File | Role |
|---|---|
| `qeft_common.py` | selection (`select_global_weak_columns`), the base builder, `QEFTLinear`, checkpoint I/O |
| `build_base.py` | stage 0 CLI — calibrate → OGR → GPTQ → dense base on disk |
| `train_commonsense.py` | stage 1 CLI — weak-column tuning on this repo's commonsense cell |
| `export_dense.py` | stage 2 CLI — scatter the trained columns back, check per layer, save |
| `smoke_test.sh` | the whole chain on a 2-layer random Llama, ~10 min on 2 GPUs |
| `../../../scripts/test_qeft.py` | unit tests: selection, `salient_ids`, `QEFTLinear`, checkpoint I/O |

Everything numerical is shared with the other rows: `src/gptq.py` quantizes, `src/permute_common.py`
supplies the calibration statistics and the permutation folds, `src/data.py` the prompt,
tokenization and calibration sampling, `runs/eval_vllm.sh` the scoring. The one change to shared
code is an **additive** `salient_ids=` argument on `gptq_quantize_model_sequential` (see o_proj
below); it is `None` for every existing caller and `scripts/test_qeft.py` asserts that passing
`None` is byte-identical to omitting it.

## The calibration — the one departure every `bcal` row makes

Upstream calibrates on c4/wikitext2. This repo measured generic calibration as the failure mode at
low bit width, not a detail: the same merged checkpoint scored **INT2 36.64** on the old first-N
in-domain set (100 % BoolQ, 9.5k tokens) and **66.22** on a task-balanced 3500-record in-domain set
(471k tokens), while C4 128×2048 loses the instruction template outright (30.67). So **both**
offline consumers move to the balanced in-domain set every `bcal` row is defined by
(`src/data.load_calibration_data`, `calibration_samples: 3500`, `calibration_sampling: balanced`,
`calibration_seq_len: 2048`, prompt template included):

* the **λ statistics** that choose the weak columns, and
* the **GPTQ Hessians** that quantize everything else.

`qat.gptq.nsamples` must equal `qat.sqat.calibration_samples`, because
`gptq_quantize_model_sequential` consumes only the first `nsamples` records.

## Choices that are not upstream defaults

* **The quantization grid is this repo's, not QEFT's.** Upstream runs a per-group grid search
  (40 candidate clipping ranges, L2.4 loss) before OPTQ; `src/gptq.py` uses the min-max group grid
  with static groups. Every other row in these tables (GPTQ floor, QA-LoRA, SALT-Q, QWHA) sits on
  that grid, and a baseline on a different grid measures the grid rather than the method. This is
  the one place where the reproduction is deliberately *weaker* than upstream, and it is worth
  quantifying separately if the row lands low: the paper's own Table 6 puts the OWQ-style search
  about 1 point of 6-task average above RTN at 4 bits.
* **The weak columns are excluded from GPTQ's error compensation as an independent sub-problem.**
  `gptq_quantize_layer` solves OBS on `H[k:, k:]`, i.e. it targets the fp16 weight of the columns
  it is allowed to move. Upstream instead puts the weak columns last and Choleskys the full
  Hessian, which makes its effective Hessian the Schur complement `A − B C⁻¹ Bᵀ`. Since the weak
  columns are exact and cannot absorb anything, the independent sub-problem is the correct
  objective; this repo's invariant (`src/gptq.py`, §"two invariants that cost real debugging time")
  says the same thing for SALT-Q's salient slice.
* **o_proj keeps its irregular weak columns where they are.** o_proj cannot be reordered — its
  input is the concatenated attention heads and mixing channels across heads is not an equivalence
  transform — which upstream handles by reordering it *online* inside its kernel. Here the
  irregular indices simply stay in place and `salient_ids` makes GPTQ quantize around them: same
  arithmetic, no gather, and honest about the fact that a packed deployment would have to ship
  those indices (upstream ships them too). `qeft.oproj_weak: false` drops them, which is what
  SALT-Q does with its salient slice.
* **Weak columns are laid out at the FRONT** of each matrix (`[0, k)`), where upstream appends
  them at the tail. Purely a convention — this repo's `group_k` machinery, its assertions and its
  fp16-salient PTQ sweep all address a leading slice — but it does mean `k` must be a multiple of
  `group_size`, so no quantization group straddles the fp16/INT boundary.
* **k = 128 at INT3 g64, k = 256 at INT2 g32.** 128 is the paper's own main-table setting (Table 2
  and 4 report k=16 and k=128, the latter chosen because `k/2 ≃ r` matches LoRA r=64's parameter
  count — 174M tunable values, which is exactly what k=128 gives on Llama-2-7B here). 256 at INT2
  matches this cell's protected-slice budget, where SALT-Q runs `group_k=256` and the rank-ordered
  fp16-salient PTQ sweep has a measured point. Matching k is what makes "same number of protected
  columns, different thing done with them" a single-variable statement against SALT-Q; carrying
  the paper's 128 into a cell whose other rows protect 256 would confound budget with method.
* **Learning rate is QEFT's own**, not this repo's: 5e-6 at k=128 (paper Table 4) and 3.5e-6 at
  k=256 (Table 7's k-sweep is a ~1/√k law: 1.4e-5 at k=8 … 0.4e-5 at k=128; one doubling from
  Table 4's value). It deliberately does **not** follow this repo's `1/(2^b − 1)` rule, because
  the trained tensors are plain fp16 weights rather than values on a quantization grid — the INT2
  grid step never enters their update, and upstream uses one lr for its 4-bit and 3-bit tables
  alike. A baseline retuned to match the method under test is not a baseline.
* **Recipe is this repo's**: 1 epoch, effective batch 80, max_seq_len 2048, cosine, warmup 0.03,
  weight decay 0.01, max_grad_norm 0.3, bf16, `group_by_length: false`, seed 42, gradient
  checkpointing on. (Upstream's Open-Platypus recipe is 1 epoch, batch 16 = 1×16, constant lr, no
  warmup, no dropout, max_grad_norm 0.3 — the step count is within 20 % of ours, so the lr
  transports without a √T correction.) The cell is the comparison; the method keeps its own knobs.
* **No PEFT merging.** The paper's §5 application (transplanting trained weak columns onto a
  different fine-tune) is orthogonal to the accuracy row and is not reproduced.

## Deployment caveat — the row is "b + fp16"

QA-LoRA and SALT-Q deploy as pure INT-b. QEFT does not, and the table must not read it as if it
did: it deploys as INT-b codes **plus k fp16 columns per linear**. `build_base.py` prints the exact
fp16 share of the target weights and the resulting effective bit width, and both are stored in
`qeft_meta.pt` and in the export meta. For Llama-2-7B, k fp16 columns cost
`k · (4·4096 + 2·11008 + 4096) = k · 42496` values per block against 202.4M target weights, so
**k=128 is 2.69 % of the target weights** (174M fp16 values — the paper's own Table 2 figure, and
within 9 % of LoRA r=64's 160M) → **3.35 effective bits** at INT3, and **k=256 is 5.37 %**
(349M) → **2.75 effective bits** at INT2. Read the row as **3 + fp16** / **2 + fp16**, next to
the fp16-salient PTQ rows.

Note this is slightly more fp16 than SALT-Q protects at the same `group_k`: SALT-Q gives o_proj
`group_k = 0` (per-head structure forbids the reordering), where QEFT keeps k weak columns there
too. o_proj is 1/7 of the projections and ~8 % of the fp16 budget; `qeft.oproj_weak: false` makes
the two budgets identical if that comparison is ever wanted.

The `--with_base` export scores the bare mixed-precision base through the same seam — INT-b + fp16
weak columns, no tuning at all. That is **this row's own floor, and it is not the fp16-salient PTQ
sweep's point at the same k**, however similar the two constructions look. The sweep quantizes an
already **QLoRA-fine-tuned merged** checkpoint (`scripts/export_mixed_precision_sweep.py`, which
starts from `outputs/qlora_*-merged-eval`), so it measures "protect k columns while deploying a
model that already knows the task". QEFT — like QA-LoRA, SALT-Q and LoTA-QAF — starts from the
**pre-trained** Llama-2-7B, so its untuned floor is a model that has never seen the instruction
format, and it will score like LoTA-QAF's own floor (8.72), not like the sweep (66–77). Read the
trained QEFT number against the other bcal rows in absolute terms; read the floor only as this
row's own starting point, i.e. as the denominator of its Gap column.

## Running it

```bash
qsub jobs/qeft_smoke.pbs                                              # harness check, 2 GPUs, ~10 min
qsub jobs/qeft_prep_int3_g64_bcal.pbs                                 # 1 GPU: OGR + GPTQ base
qsub -W depend=afterok:<prep-id> jobs/cs_qeft_int3_g64_span_bcal.pbs  # 4 GPUs: train+export+eval
```

or directly:

```bash
bash runs/qeft/run_qeft_commonsense.sh --bits 3 --group_size 64 --with_base
bash runs/qeft/run_qeft_commonsense.sh --bits 2 --group_size 32 --with_base \
    --config baseline/QEFT/sqat/configs/qeft_cs170k_int2_g32_ep1_span_bcal.yaml
```

The pipeline is `runs/qeft/run_qeft_commonsense.sh` → `runs/qeft/_pipeline.sh`, the same shape as
`runs/saltq/_pipeline.sh`, `runs/lota/_pipeline.sh` and `runs/qwha/_pipeline.sh`: base → train →
export → eval → `results_saltq.csv` (`--filter qeft`), all into one log under
`logs/commonsense_170k/`.

Every stage is idempotent — it skips when its output exists — so a job that hits the 24 h walltime
can simply be resubmitted. Artefacts land in `outputs/qeft_bases/` (the mixed-precision base),
`outputs/qeft_cs170k_*` (trained weak columns only, ~0.7 GB) and `outputs/qeft_cs170k_*-Nbit-qeft-dense-eval`
(what vLLM reads). Scores land in `results/commonsense_vllm/<tag>.json` like every other row.

**The base is a one-shot discrete choice.** Rebuilding it re-runs the calibration and can move the
weak-column SET, which would leave every trained tensor pointing at different channels;
`build_base.py` therefore refuses to overwrite a base whose settings differ from the config, and
refuses to reuse one that does not match. Do not delete `outputs/qeft_bases/` while a checkpoint
that references it still matters.
