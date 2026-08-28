# SALT-Q — results summary (Commonsense-170k main, MetaMath secondary)

All commonsense numbers are generative accuracy (vLLM greedy, `runs/eval_vllm.sh`) on the eight
LLM-Adapters test sets, Llama-2-7B, 1 epoch unless stated; **MEAN(8)** is the headline, MEAN(7)
drops openbookqa (n=500). Every row is one seed; pairwise resolution is about ±0.5 on MEAN(8).
Per-task numbers and the run notes are in `results_saltq.csv`.

## Main result: Commonsense-170k, 1 epoch, `data.loss_span = instruction+response`

The supervision span is a property of the cell: question + options + response are supervised
(the fixed prompt header masked, ~133 supervised tokens/record). It is what removed the INT2
plateau pathologies; its price is a ~4.5-point ceiling drop that hits every method alike, so rows
are comparable only within a cell. Bounds: the same QLoRA run merged to fp16 (upper) and
merged-then-GPTQ at the cell's width (floor).

| INT3 g64 | eff. bits | MEAN(8) | MEAN(7) | gap recovered |
|---|---|---|---|---|
| QLoRA merged fp16 (upper bound) | 16 | 77.75 | 77.72 | 100% |
| **SALT-Q** (k=128, salient_lr 5e-5) | **3.00** | **77.76** | **78.02** | **100%** |
| QA-LoRA | 3.00 | 76.82 | 76.76 | 82% |
| fp16-salient PTQ, best k (2048, 39% fp16) | 8.05 | 74.72 | — | 41% |
| QLoRA merged → GPTQ (floor) | 3.00 | 72.60 | 72.75 | 0% |

| INT2 g32 | eff. bits | MEAN(8) | MEAN(7) | gap recovered |
|---|---|---|---|---|
| QLoRA merged fp16 (upper bound) | 16 | 77.75 | 77.69 | 100% |
| **SALT-Q** (k=256, zp_lr 1.73e-3, salient_lr 1.25e-4) | **2.00** | **69.61** | **69.64** | **80%** |
| SALT-Q (k=128, same lrs) | 2.00 | 67.16 | 67.29 | 74% |
| QA-LoRA | 2.00 | 67.11 | 67.64 | 74% |
| fp16-salient PTQ, best k (2048, 39% fp16) | 7.44 | 58.96 | — | 54% |
| QLoRA merged → GPTQ (floor) | 2.00 | 36.64 | 38.08 | 0% |

Figures (fp16-salient PTQ swept over rank-ordered top-k, x = true fp16 share, MEAN(8)):
`figures/mixedprec_int2_g32_span_rank_mean8_share.png`,
`figures/mixedprec_int3_g64_span_rank_mean8_share.png`.

Reading: at INT3, SALT-Q at 3.00 bits sits on the cell's fp16 ceiling while the fp16-salient
PTQ curve is flat from k=64 on (top-64 by saliency already gives everything the fp16 slice
gives). At INT2, k=256 is the knee of the PTQ curve (44.6 → 49.2) and SALT-Q trained at k=256
turns it into +2.45 over k=128 at the same 2.00 deployed bits; PTQ needs 7.4 effective bits to
reach 59. SALT-Q's INT2 lr peak is bracketed at the values above (0.5× 66.17 / 1× 67.29 / 2.5×
66.24 MEAN(7)).

Longer training (secondary, same cell): SALT-Q k=128 at 3 epochs, lrs unchanged → **70.61 /
70.73**; every span-cell run is still descending at the end of epoch 1. A 3-epoch SALT-Q is not
comparable to the 1-epoch QA-LoRA row.

## Reference: Commonsense-170k, 1 epoch, response-only supervision (the original cell)

| | INT3 g64 MEAN(8) / (7) | INT2 g32 MEAN(8) / (7) |
|---|---|---|
| QLoRA merged fp16 (upper) | 82.72 / 82.42 | 82.72 / 82.42 |
| SALT-Q (k=128) | 81.62 / 81.48 | 64.68 / 65.77 (3 ep: 72.24 / 72.24) |
| QA-LoRA | 81.08 / 80.92 | 71.26 / 71.90 |
| QLoRA merged → GPTQ (floor) | 78.26 / 78.38 | 50.01 / 50.35 |

INT3 is additive and SALT-Q leads (+0.56 MEAN(7), within one-seed resolution). INT2 under
response-only supervision has a training plateau (one informative token per record) that
full-rank zero-point SGD escapes late or never; that pathology, not capacity, is what the span
cell removes (see the branch history for the decomposition experiments).

## Secondary result: MetaMath → GSM8K (INT2 g32, 1 epoch, flexible-extract)

| | GSM8K |
|---|---|
| SALT-Q, AWQ fold + salient reorder (legacy [2,30] / k=128) | **40.94** |
| SALT-Q (autoseg k=128, zp_lr 5e-3) | 37.67 |
| QA-LoRA (lr 2e-3, tuned) | 37.38 |
| QA-LoRA (lr 1e-4, paper) | 19.18 |
| SALT-Q z-only ablation | 29.95 |

INT3 g64 MetaMath: SALT-Q 45.11 (no QA-LoRA / QLoRA bounds were run at INT3 on math).

## What is kept on disk

`outputs/` (→ `/scratch/.../SQAT_outputs`) now holds only the runs in the two main tables plus
their bases: for each width the SALT-Q run(s), the QA-LoRA run, the QLoRA run with its merged
and GPTQ exports, the shared permuted/GPTQ bases, and the rank-ordered permutation meta
(`mixedprec_rankperm_k2048/sqat_permute_meta.pt`). Every deleted run's scores are in
`results/` and `results_saltq.csv`; its config and job are in git.
