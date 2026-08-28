# LoTA-QAF baseline — what is upstream, what is ours, and why

Upstream: <https://github.com/KingdalfGoodman/LoTA-QAF> (MIT), cloned 2026-08-28,
`LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning`, NeurIPS'25,
arXiv:2505.18724.

Everything at the top level of this directory except the files listed under **Ours** is the
upstream tree, unmodified. `vendor/` holds pinned checkouts of the two libraries LoTA-QAF
patches rather than imports (see `setup_env.sh`); they are not tracked here.

## Ours

| File | Why it exists |
|---|---|
| `setup_env.sh` | Builds the isolated `lota` env (torch 2.6 / peft 0.15.1 / gptqmodel 2.1.1-dev). The SQAT training env runs torch 2.11 + peft 0.20 and every checkpoint in `outputs/` was produced with it. |
| `patches/apply_lota_patches.py` | `LoTA/layer.py` is a *fragment* to be spliced into `peft/tuners/lora/layer.py`; the two surrounding edits (`model.py` must forward `custom_config`, `other.py` must treat the model as GPTQ-quantized) are described only in upstream prose. This applies all three mechanically. |
| `quantize_base.py` | Upstream `gptq_quantize.py` with the `/your_path/` placeholders turned into flags. Recipe unchanged. |
| `train_lota.py` | Upstream `LoTA_QAF_main.py` mode 1, with this repo's data recipe and the real optimizer-step count for the sigma schedule (below). |
| `export_lota_dense.py` | Merges the ternary adaptation into the integer codes exactly as upstream's `LTA.merge()` does, then writes a dense fp16 checkpoint so the shared evaluation can score it. |
| `tests/test_export_consistency.py` | Asserts the exported weight reproduces the trained `CustomLoraLinear` forward. |
| `configs/`, `run_lota_commonsense.sh`, `jobs/` | The two cells (INT3 g64, INT2 g32) and the pipeline that produces them. |

## The five deviations, and why each one is required

1. **Data.** Upstream `prepare_dataset()` covers alpaca/gsm8k/sql/viggo and renders every
   record through `tokenizer.apply_chat_template`. Llama-2-7b-hf is a base model with no chat
   template, and the comparison is on Commonsense-170k. `train_lota.py` therefore calls
   `src/data.py` — the same PROMPT, the same `loss_span = instruction+response` masking, the
   same collator and shuffle seed as the SALT-Q and QA-LoRA rows. Anything else would have
   compared a prompt as well as a method.

2. **sigma_t step count.** Upstream hardcodes the schedule length per dataset
   (`{"alpaca": 300, "gsm8k": 117, ...}`). The anneal is defined in *fractions* of the run
   (top 5% → 0.1% over the first 80%, then 0.01%), so on a dataset that is not in that dict the
   schedule would land in the wrong place. `train_lota.py` passes the run's real optimizer-step
   count. This is the closest thing t-SignSGD has to a learning-rate schedule.

3. **Evaluation.** Upstream evaluates through GPTQModel's kernels with an `LTA` adapter object
   under lm-eval (MMLU) or its own `evalGSV.py`. Every row of `RESULTS_SUMMARY.md` instead comes
   from `runs/eval_vllm.sh`: greedy vLLM generation on `datasets/commonsense/test.json`, exact
   match through `scripts/test_acc.py`. Scoring this baseline its own way would put a different
   harness and metric next to our numbers, so the merge is done offline
   (`export_lota_dense.py`) and the result is scored by the shared harness. The merge is the
   same arithmetic as `LoTA/lota_merge.py`, and `tests/test_export_consistency.py` checks it
   against the live training forward rather than assuming it.

   Consequently `LoTA/adapter.py` and `LoTA/lota_merge.py` (the gptqmodel-side inference
   patches) are **not** applied — nothing in this reproduction runs them.

4. **bf16 gradients in t-SignSGD.** `torch.quantile()` accepts float/double only, and the
   ternary adapters are bf16 (`IntLinear` forces it), so `torch.quantile(p.grad.abs(), ...)`
   raises on this stack. The quantile is computed on a float copy and the resulting threshold
   compared against the original magnitudes — an identity when gradients are fp32. The same
   commit casts `check_grad_norm`'s accumulator, where a bf16 sum of ~10^5 squares would drop
   the small terms and make the norm an artifact of the accumulation dtype. Committed as a
   diff against the verbatim upstream tree.

5. **Micro-batch split.** The effective batch is 80, matching the other rows. Upstream runs one
   GPU at batch 64; we run one GPU at 20 x 4 accumulation. Gradient accumulation sums into the
   same optimizer step, and LoTA-QAF's per-step cost is dominated by weight-side work (the
   `[in, out]` ternary product, the threshold kernel, the dense reconstruction) that does not
   scale with tokens in the batch.

## What is deliberately NOT matched to our cell

* **The GPTQ base.** Built by GPTQModel with the paper's recipe: asymmetric, `desc_act` on,
  damp 0.01, 1024 C4 sequences — not `src/gptq.py` calibrated on the training data. LoTA-QAF
  moves each weight by at most ±1 step *on the grid it is handed*, so that grid is part of the
  method. The cost is that our GPTQ floor row is not this method's floor, so the bare base is
  exported and scored too (`--with_base`, reported as *LoTA-QAF base (GPTQModel)*).
* **omega = 0.75r = 48 and sigma_t 5% → 0.1% → 0.01%**, the paper's Section 4.1 values, and
  `max_grad_norm` disabled. Same stance as `configs/qalora_*.yaml`: a baseline retuned to match
  the method under test is not a baseline.

## Deployment width

The merged model is `scales[g] * (q' - zeros[g] + mu_g[g])`: integer codes at the stated bit
width plus a per-group fractional zero-point shift. That is the same deployment contract as the
QA-LoRA rows (a group-constant delta folded into the affine zero-point), so "2 bits" means the
same thing in both tables. The dense fp16 file is a container for the evaluation harness, not a
wider format.


## What the INT3 g64 run measured (2026-08-29)

First completed cell: Commonsense-170k, 1 epoch, `loss_span=instruction+response`, omega=48,
sigma_t 5%->0.1%->0.01%, the paper's own settings.

| Row | Avg (MEAN8) |
|---|---|
| LoTA-QAF | **53.13** |
| LoTA-QAF base (GPTQModel, raw) | **9.63** |

Read it against the same-span SALT-Q (77.76) and QA-LoRA (76.82), and against **9.63**, not
against this repo's 72.60 GPTQ floor: that row is QLoRA-fine-tuned and *then* quantized, so it
already knows the task, whereas LoTA-QAF's base is raw pretrained Llama-2 and cannot emit the
answer format at all. Gap recovered inside its own bracket is (53.13-9.63)/(77.75-9.63) = 64%.
The honest head-to-head is 53.13 vs QA-LoRA's 76.82: both start from a raw quantized base and
see the same data, batch and epoch.

Three measurements explain the shape of that result, all taken from the trained adapter:

1. **The ternary adapter trained hard.** `lora_B` is zero-initialised by construction; after
   1844 steps both A and B are dense ternary (~1/3 each of -1, 0, +1). t-SignSGD is working.

2. **The integer merge is nearly inert.** |AB| runs p50=6, p90=19, p99=34, p99.9=45, max=61
   against omega=48, so 3.4e-04 of weights cross the threshold in layer 0's down_proj and
   *zero* cross it in layer 23's. The +-1 grid-step moves the method is named for barely
   happen here: with rank r, |AB| is a sum of r ternary products and has std sqrt(4r/9) = 5.3
   at r=64, so omega = 0.75r is a ~9-sigma threshold that training only reaches in the far tail.

3. **So all 43.5 points of lift come from the offset factor** -- and the released code scales
   that offset by 1/omega where the paper does not. `layer.py:359`, `adapter.py:391` and
   `lota_merge.py:382` all compute `groupmean(dW - omega*markers) / self.threshold`, while
   Eq. (4)-(5) define `mu = mean(W~)` and `z' = z + s*mu` with no such division. Measured on
   this checkpoint, that is a deployed zero-point shift of |mu| = 0.022 grid steps (p99 0.098)
   under the code, against 1.06 (p99 4.72) under the paper's formula -- a factor of 48.

Because the division is consistent across upstream's training, inference and merge paths it is
not a train/deploy inconsistency, and this reproduction follows the released code, which is
what reproducing an official implementation means. But it is the obvious candidate for the
23.7-point gap to QA-LoRA, whose group-pooled delta folds into the zero-points unscaled. Note
also that the paper's own INT3/INT4 *performance-recovery* deltas (+0.07 / +0.03 MMLU) are
consistent with a small offset, while its INT2 (+15.45) and task-specific numbers are not
obviously reachable with one.

An arm using Eq. (4)'s scaling is a one-line change and would separate "the method is weak on
this task" from "the released scaling is not what produced the published numbers". Not run.
