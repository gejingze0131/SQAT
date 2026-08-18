# QLoRA + Selective Salient QAT Experiment Framework

## Two conda envs, on purpose

| env | built by | holds | used for |
|-----|----------|-------|----------|
| `saltq` | `requirements.txt` | torch / transformers / peft / bitsandbytes | training, the offline permute + GPTQ pre-steps, export |
| `vllm` | `scripts/setup_vllm_env.sh` (`requirements-vllm.txt`) | vLLM and its own pinned torch | generative evaluation |

They are kept apart because vLLM pins its own torch build, and a quantization method is
exactly the kind of code whose numbers move when the numerics under it move. The two envs
never have to agree on anything — they exchange a plain HF checkpoint on disk.
`run_eval_vllm.sh` is the seam and hops between them itself, so a PBS job stays in `saltq`.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
bash scripts/setup_vllm_env.sh          # one-off
```

## Data

Training and test data live under `datasets/<name>/{train,test}.json` (gitignored), in the
pissa-dataset schema — a flat JSON array of `instruction` / `input` / `output` / `type`:

* `datasets/commonsense` — 147k train / 22k test over BoolQ, PIQA, SIQA, ARC-e/c, HellaSwag,
  WinoGrande, OBQA
* `datasets/metamath` — 395k MetaMathQA train / 6.3k GSM8K + MATH test

`src/data.py` renders each record through one prompt (`src/data.PROMPT`) and supervises the
**response only** — the prompt span is masked with `IGNORE_INDEX` and the target ends in an
explicit EOS. The test splits ship their `instruction` already wrapped in that same prompt, so
`scripts/gen_vllm.py` feeds it to the model verbatim.

## Quick start

```bash
# Train (saltq env). Eval is stage 3 of the same script and switches env by itself.
bash run_saltq.sh --config configs/saltq_cs170k_int3_g64.yaml --bits 3 --dataset commonsense

# Baselines, same data and same metric
bash run_none_qlora.sh --dataset commonsense --bits 3      # QLoRA
bash run_qalora.sh     --dataset commonsense --bits 3      # QA-LoRA

# Evaluate an already-exported model on its own
bash run_eval_vllm.sh --model_path outputs/saltq-3bit-saltq-deploy-eval --dataset commonsense
```

Results land in `results/<dataset>_vllm/<tag>.jsonl` (raw generations) and `<tag>.json`
(per-task accuracy), and are folded into `results_saltq.csv` by
`scripts/collect_saltq_results.py`.

## Evaluating a permuted export

SALT-Q and SQAT-Permute exports are **not** self-contained Llamas: their residual stream is
permuted per segment and a correct forward pass needs a `BoundaryGatherHook` at each segment
boundary. vLLM cannot register that hook, so `scripts/export_vllm_ready.py` folds the
permutation back into the weights first (an exact reindex — `scripts/test_unpermute_fold.py`
checks it against a reference forward pass). `run_eval_vllm.sh` does this automatically;
`scripts/gen_vllm.py` refuses to run on a directory that still carries
`sqat_permute_meta.pt`.

## Layout

```
configs/                 experiment configs (one per cell; the comments carry the rationale)
src/
  model_loader.py        model + tokenizer loading (NF4/NF3 + LoRA)
  data.py                prompt, response-only-loss tokenization, collator
  trainer.py             HF Trainer + QAT callbacks, per-tier optimizer groups
  qat_base.py            QAT interface + Full QAT
  qat_sqat.py            Selective Salient QAT
  qat_permute_sqat.py    segment-permuted SQAT
  qat_saltq.py           SALT-Q
  qalora.py              QA-LoRA baseline
  permute_common.py      offline permutation, boundary gathers, and the fold that removes them
  export.py              merge / dequantize / export
scripts/
  train.py               main training entry
  gen_vllm.py            generative eval, stage 1 (vllm env)
  test_acc.py            generative eval, stage 2 — exact-match scoring
  export_vllm_ready.py   fold the residual permutation out of an export
  setup_vllm_env.sh      build the eval env
  test_*.py              correctness checks
run_saltq.sh             SALT-Q: train -> export -> eval
run_none_qlora.sh        QLoRA baseline
run_qalora.sh            QA-LoRA baseline
run_eval_vllm.sh         eval only, in the vllm env
```
