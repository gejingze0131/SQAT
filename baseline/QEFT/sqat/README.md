# QEFT baseline — how to run it

Reproduces **QEFT** (*Quantization for Efficient Fine-Tuning of LLMs*, Findings of EMNLP 2024,
[arXiv:2410.08661](https://arxiv.org/abs/2410.08661), upstream
[xvyaward/qeft](https://github.com/xvyaward/qeft)) on Llama-2-7B / Commonsense-170k, inside this
repo's experimental cell. What is upstream and what is ours — and why each departure exists — is
in [PROVENANCE.md](PROVENANCE.md). This file is only about **running it**.

## No new environment

QEFT runs in the repo's own `saltq` env. Unlike `baseline/LoTA-QAF/` and `baseline/QWHA/`, nothing
upstream is vendored: QEFT's repository is a fork of OWQ built around CUDA kernels for its packed
mixed-precision format, and this table scores every row by generating with vLLM from a **dense**
checkpoint, so those kernels buy inference speed that is not being measured. The method itself —
Hessian-diagonal weak-column selection, one global reordering, GPTQ that skips the weak columns,
and training only those columns — is implemented against this repo's `src/gptq.py`,
`src/permute_common.py` and `src/data.py`.

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate saltq
```

## Check the harness first (~10 min, 2 GPUs)

```bash
qsub jobs/qeft_smoke.pbs                    # or: bash baseline/QEFT/sqat/smoke_test.sh
python scripts/test_qeft.py                 # unit tests alone, CPU, seconds on a compute node
```

The smoke test runs every stage the 7B run does — mixed-precision base, DDP weak-column tuning,
dense export with its per-layer equivalence check — on a 2-layer random Llama, and prints
`SMOKE TEST OK`.

## Run it

```bash
# 1 GPU, ~1 h: calibrate λ, pick the global weak columns, fold the permutation, GPTQ the rest
qsub jobs/qeft_prep_int3_g64_bcal.pbs
qsub jobs/qeft_prep_int2_g32_bcal.pbs

# 4 GPUs: train -> dense export -> vLLM eval -> results_saltq.csv, all into one log
qsub -W depend=afterok:<prep-id> jobs/cs_qeft_int3_g64_span_bcal.pbs
qsub -W depend=afterok:<prep-id> jobs/cs_qeft_int2_g32_span_bcal.pbs
```

or directly:

```bash
bash runs/qeft/run_qeft_commonsense.sh --bits 3 --group_size 64 --with_base
bash runs/qeft/run_qeft_commonsense.sh --bits 2 --group_size 32 --with_base \
    --config baseline/QEFT/sqat/configs/qeft_cs170k_int2_g32_ep1_span_bcal.yaml
```

Useful flags: `--skip_base`, `--skip_train`, `--skip_export`, `--skip_eval`, `--num_gpus`,
`--resume_from <checkpoint>`, `--with_base` (also score the untuned base — this row's own floor),
`--note "..."` (goes into `results_saltq.csv`).

## What lands where

| | |
|---|---|
| `outputs/qeft_bases/Llama-2-7B_int{b}_g{gs}_k{k}_asym_bcal` | the mixed-precision base + `qeft_meta.pt` (weak-column indices, fp16 share, effective bits) |
| `outputs/qeft_cs170k_*/final` | trained weak columns ONLY (~0.7 GB) + a pointer to the base |
| `outputs/qeft_cs170k_*-{b}bit-qeft-dense-eval` | the dense checkpoint vLLM reads |
| `results/commonsense_vllm/<tag>.{jsonl,json}` | generations and per-task accuracy |
| `results_saltq.csv` | the shared table (`--filter qeft`; methods `QEFT` and `QEFT base (bcal GPTQ + fp16 weak)`) |
| `logs/commonsense_170k/qeft_cs170k_*.log` | one log per run, all stages |

Every stage is idempotent — it skips when its output exists — so a job that hits the walltime can
be resubmitted. **Do not delete a base while a checkpoint that references it still matters:** the
weak-column set is a one-shot discrete choice, and a rebuilt base would leave the trained columns
pointing at different channels. `build_base.py` refuses to reuse a base whose settings disagree
with the config, and refuses to overwrite one without `--force`.

## Cost, on A100-40GB

| stage | GPUs | time | notes |
|---|---|---|---|
| base | 1 | ~1 h | dominated by GPTQ over 3500 calibration records; ~30 GB host RAM |
| train | 4 | ~5 h | 1 epoch of 147k records at effective batch 80 |
| export | 1 | ~5 min | a scatter plus a per-layer check |
| eval | 4 | ~40 min | vLLM, 22k generations |

Training holds one dense bf16 base per rank (~13 GB) plus fp32 master weights, gradients and Adam
state for the k weak columns only (174M params at k=128 → ~2.8 GB). Gradient checkpointing is on
by default, as it is in the paper's own measurements.

## Reading the result

The row is **not** a pure INT-b row: QEFT deploys as INT-b codes plus k fp16 columns per linear.
The exact fp16 share and effective bit width are printed by `build_base.py` and stored in the
export meta — quote them next to the accuracy, and read the row against the other `bcal` rows, not
against the pure-INT rows.

`--with_base` scores the untuned base as this row's own floor. It is **not** the fp16-salient PTQ
sweep's point at the same k: that sweep quantizes an already fine-tuned merged checkpoint, while
QEFT starts from the pre-trained model, so its floor is a model that has never seen the
instruction format (expect LoTA-QAF-floor territory, ~9, not 66–77). See PROVENANCE.md
§"Deployment caveat".
