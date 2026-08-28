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
