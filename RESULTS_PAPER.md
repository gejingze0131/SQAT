# SALT-Q — paper-ready results

Llama-2-7B, Commonsense-170k, 8-task generative suite scored through vLLM (greedy, 22,419 test
prompts). Single seed (42) throughout.

**Reporting conventions used below, and why.**

- **MEAN(7) excludes OpenBookQA.** It has only 500 test examples (stderr 2.4 vs 0.5–1.5 for the
  rest) and it has already carried an entire apparent win on its own. MEAN(8) is given alongside
  for completeness.
- **Pairwise resolution at one seed is ±0.48 MEAN(7).** Any difference below ~0.95 is not
  readable from a single run, no matter how clean the table looks.
- Every non-fp16 row is a **real deployed model at the stated bit width**, exported and scored on
  its own generations. The PTQ floor is quantized through the *same* error-compensated
  `gptq_quantize_model_sequential` that builds SALT-Q's frozen base and QA-LoRA's base, so no row
  is flattered by facing an RTN baseline.
- Controls held identical across methods: dataset, prompt, loss masking, collator, effective batch
  80, seed 42, 1 epoch (T = 1845 steps), `group_by_length: false`. **QA-LoRA keeps its own
  published learning rate (5e-3)** — a baseline retuned to match the method is not a baseline.

---

## Table 1 — Main result, INT3 g64, 1 epoch

Every row except fp16 is deployed at **3.00 bits**.

| method | ARC-c | ARC-e | OBQA | BoolQ | PIQA | SIQA | HellaS | WinoG | MEAN(8) | **MEAN(7)** |
|---|---|---|---|---|---|---|---|---|---|---|
| fp16 LoRA merge *(upper bound)* | 73.98 | 88.05 | 84.80 | 70.52 | 84.06 | 81.58 | 94.56 | 84.21 | 82.72 | **82.42** ±0.33 |
| **SALT-Q (ours), k=128** | 70.73 | 85.98 | 82.60 | **71.41** | **84.11** | 81.06 | 93.55 | 83.50 | 81.62 | **81.48** ±0.34 |
| QA-LoRA | 70.73 | 85.73 | 82.20 | 70.28 | 83.30 | 80.86 | 93.53 | 82.00 | 81.08 | 80.92 ±0.34 |
| Permuted-SQAT (LoRA + frozen GPTQ) | 70.56 | 83.80 | 82.40 | 68.23 | 81.66 | 79.53 | 91.99 | 79.24 | 79.68 | 79.29 ±0.35 |
| QLoRA → GPTQ *(PTQ floor)* | 67.32 | 83.12 | 77.40 | 68.69 | 78.51 | 79.38 | 91.38 | 80.27 | 78.26 | 78.38 ±0.36 |

**Significance.**

| comparison | Δ MEAN(7) | z | tasks individually sig. | sign test |
|---|---|---|---|---|
| SALT-Q − PTQ floor | **+3.10** | **6.32** | 5/7 | 7/7, p=0.0078 |
| SALT-Q − Permuted-SQAT | **+2.19** | **4.51** | 5/7 | 7/7, p=0.0078 |
| QA-LoRA − PTQ floor | +2.54 | 5.15 | 3/7 | 7/7, p=0.0078 |
| fp16 − SALT-Q | +0.94 | 2.00 | 2/7 | 5/7, p=0.2266 |
| **SALT-Q − QA-LoRA** | **+0.56** | **1.17** | **0/7** | 6/6 (1 tie), p=0.0156 |

SALT-Q closes **76.7%** of the 4.04-point quantization gap; QA-LoRA closes 62.8%.

> **What can and cannot be claimed.** "SALT-Q beats the PTQ floor and beats the LoRA-form variant"
> is solid — large margins, majority of tasks individually significant. **"SALT-Q beats QA-LoRA"
> is NOT yet claimable**: +0.56 against a ±0.48 resolution, with zero tasks individually
> significant. What supports it is only the *consistency* — SALT-Q is ahead on every non-tied task
> (sign p=0.0156) and the same ordering replicates at 3 epochs (Table 2). Report it as
> "consistently but marginally ahead", or run ≥3 seeds before making it a headline (3 seeds →
> ±0.27, 5 seeds → ±0.21).

---

## Table 2 — INT3 g64, 3 epochs (replication)

| method | MEAN(8) | MEAN(7) |
|---|---|---|
| fp16 LoRA merge | 84.09 | 83.85 ±0.32 |
| **SALT-Q (ours), k=128** | 82.20 | **82.06** ±0.33 |
| QA-LoRA | 81.70 | 81.57 ±0.34 |
| QLoRA → GPTQ (PTQ floor) | 80.62 | 80.40 ±0.34 |

SALT-Q − QA-LoRA = **+0.48** (z=1.02, 0/7 sig, 6/6 wins with 1 tie, sign p=0.0156) — the same
picture as 1 epoch, which is the strongest available evidence that the ordering is real.

**Report this honestly:** gap recovery is *worse* at 3 epochs (**48.1%**) than at 1 epoch (76.7%),
because the fp16 bound climbs faster (82.42 → 83.85) than SALT-Q does (81.48 → 82.06). More
training does not narrow the quantization gap here; it widens it.

---

## Table 3 — Ablations, INT3 g64, 1 epoch

**Salient budget `group_k`** — a clean negative, and worth reporting as one. `group_k` costs **zero
deployment bits** (the salient slice is quantized at export like everything else), so this is a
pure trainability knob.

| group_k | share of target weights | MEAN(7) |
|---|---|---|
| 64 | 1.21% | 81.09 ±0.34 |
| **128** (default) | 2.43% | **81.48** ±0.34 |
| 256 | 4.86% | 81.61 ±0.33 |

0.52 spread over a 4× range, entirely inside noise: **protection saturates almost immediately**.
Independently confirmed by the mixed-precision sweep (Table 5), which is flat from k=128 to
k=1024. Outlier energy is concentrated.

**Component decomposition** — measured from the PTQ floor, this is the paper's mechanism result.

| variant | Δ vs floor | z | sign test | verdict |
|---|---|---|---|---|
| **trainable (s, z) alone** (z-only) | **+2.15** | **4.34** | 7/7, p=0.0078 | **significant** |
| LoRA form alone (Permuted-SQAT) | +0.91 | 1.82 | 5/7, p=0.2266 | **not distinguishable from noise** |
| salient QAT added on top of z-only | +0.95 | 1.98 | 6/7, p=0.0625 | borderline, consistent |
| full SALT-Q | +3.10 | 6.32 | 7/7, p=0.0078 | significant |

**Claim this:** what carries the method is whether the adaptation **survives into the deployed
quantization parameters** — not the rank of the update and not the LoRA form. Trainable zero-points
alone recover 2.4× what the LoRA form recovers, and the LoRA form alone is statistically
indistinguishable from doing nothing. Low-rank vs full-rank `z` was separately measured at −0.39
(z=0.81, noise), so rank is *not* the axis.

**Salient learning rate** — sharply unimodal; worth a figure, since it shows the method needs a
*small* salient displacement (0.041 grid steps, twelve times below the value transferred from
MetaMath).

| salient_lr | \|ΔW_S\| (grid steps) | MEAN(7) |
|---|---|---|
| 0 (z-only) | — | 80.53 |
| **5.0e-5** | **0.041** | **81.48** |
| 1.0e-4 | 0.079 | 80.98 |
| 2.0e-4 | 0.158 | 80.26 |
| 3.46e-4 | 0.273 | 78.76 |

---

## Table 4 — INT2 g32, 1 epoch: **negative result, must be reported**

| method | MEAN(8) | MEAN(7) | degenerate tasks |
|---|---|---|---|
| QA-LoRA | 71.26 | **71.90** ±0.38 | none |
| **SALT-Q (ours), k=128** | 64.68 | **65.77** ±0.40 | BoolQ collapsed |
| QLoRA → GPTQ (PTQ floor) | 50.01 | 50.35 ±0.42 | BoolQ, HellaSwag below majority |

**QA-LoRA beats SALT-Q by 6.13 (z=11.0, 7/7 tasks individually significant).** This is unambiguous
and cannot be presented as noise. SALT-Q still recovers +15.42 over the floor (z=26.6), and the
floor at 2 bits is a destroyed model (HellaSwag 24.4, below chance), but that does not rescue the
comparison.

**Mechanism, measured — this is what makes the negative result publishable rather than embarrassing.**
At INT2 every method parks on a loss plateau near 0.13 and they differ only in **when they leave
it**, and escape epoch predicts the score monotonically:

| variant | escape epoch | MEAN(7) |
|---|---|---|
| QA-LoRA (low-rank → update *must* concentrate) | **0.347** | 71.90 |
| SALT-Q (full-rank z + trainable salient weights) | 0.672 | 65.77 |
| SALT-Q z-only (full-rank z only → maximally uniform) | 0.938 | 48.54 |
| SALT-Q, zp_lr ×5 (uniform update, scaled up) | **never** | 38.28 (all 8 tasks degenerate) |

Measured in the shared mean-pooled coefficient space, the two methods do **not** differ in how much
they adapt — at the median SALT-Q moves *further* (1.125e-2 vs 9.950e-3 at INT2) — they differ in
**how concentrated** the adaptation is: p99/p50 is **15.3 for QA-LoRA vs 5.7 for SALT-Q**, i.e.
SALT-Q's update is 2.7× flatter. Rank ≤ 64 on an `[out, G]` matrix is a sum of 64 outer products
and *must* pile onto a few directions; full-rank `z` under Adam gets per-parameter normalization,
so every coefficient takes about the same small step and nothing can concentrate.

**Interpretation for the paper:** full-rank zero-point adaptation is an *advantage* where the
frozen codes are good enough that uniform small corrections suffice (INT3, where SALT-Q wins), and
a *liability* where escaping a bad initialization needs a few large targeted corrections (INT2).
Two things confirm the direction rather than merely being consistent with it: raising `zp_lr` — which
scales every coefficient equally and adds no concentration — made the run *collapse entirely*, and
removing the salient tier, the one component that *can* move a few columns a lot, pushed the escape
*later* and cost 17.23 points.

Ruled out as explanations, with evidence: the zero-point clamp (`0.00%` of groups at either
boundary of `[0,3]`); `group_k` (QA-LoRA's INT2 base has **no** protected columns at all and still
wins); `salient_lr` scaling (`|ΔW_S|` = 0.041 grid steps at both bit widths, exactly as designed).

---

## Table 5 — Mixed precision: does training replace keeping the salient slice in fp16?

The alternative SALT-Q claims to replace, measured training-free: take the fp16 QLoRA merge (82.42,
task adaptation already in the weights), replay SALT-Q's own permutation, keep columns `[0:k)` in
**fp16** and GPTQ the rest at INT3 g64 (`keep_salient_fp16=True`, same quantizer, same salient
channels, same segmentation). Gap to close = 4.04 points.

| k | fp16 share | **effective bits** | MEAN(7) | gap recovered | SALT-Q(3.00 bit) − this |
|---|---|---|---|---|---|
| 0 | 0% | 3.00 | 78.38 | 0% | +3.10 (z=6.32) |
| 64 | 1.21% | 3.16 | 79.35 | 24.0% | +2.13 (z=4.38, 7/7, p=0.0078) |
| 128 | 2.43% | 3.32 | 80.30 | 47.4% | +1.18 (z=2.46, 6/7) |
| 256 | 4.86% | 3.63 | 79.97 | 39.4% | +1.51 (z=3.11, 7/7, p=0.0078) |
| 512 | 9.72% | 4.26 | 80.33 | 48.2% | +1.15 (z=2.39, 7/7, p=0.0078) |
| 1024 | 19.43% | 5.53 | 80.56 | 53.9% | +0.92 (z=1.92, 6/7) |
| 2048 | 38.86% | 8.05 | 81.18 | 69.3% | +0.30 (z=0.63, **tied**) |
| — | 100% | 16.00 | 82.42 | 100% | −0.94 |
| **SALT-Q k=128** | — | **3.00** | **81.48** | **76.7%** | — |

**Headline claim, and it is well supported:** a real 3.00-bit trained model matches what
mixed-precision protection needs **5.5–8 effective bits** to reach, and strictly beats every point
at or below 4.26 bits. **Training the salient slice is worth roughly 2.5–5 bits** against spending
precision on it. The matched point is the cleanest statement: at k=128 both arms protect the *same*
2.43% of columns under the *same* segmentation, and SALT-Q is +1.18 while using **fewer** bits.

**Limitation to state, not hide:** this arm is **PTQ** — QLoRA-trained, then quantized, with no
adaptation to the quantizer in the loop. So it answers "how many bits must you spend to skip the
training", which is the claim being made, but it is **not a QAT-vs-QAT comparison**. QA-LoRA
remains the quantization-aware baseline, and that gap is still the unresolved +0.56.

Figure: `figures/mixedprec_curve.png` (`scripts/plot_mixedprec_curve.py`).

---

## What is missing before submission

1. **Multi-seed on the SALT-Q vs QA-LoRA cell.** This is the only blocking gap. +0.56 at ±0.48 is
   not a result; 3 seeds gives ±0.27 and 5 gives ±0.21. The 6/6 sign test replicating at both 1
   and 3 epochs says the ordering is probably real, which is exactly why it is worth the seeds.
2. **A second model** (Llama-2-13B or Mistral-7B). Every number here is one model.
3. **A second task family.** MetaMath results exist from earlier work but were not re-run on this
   pipeline, so they are not directly comparable to anything above.
4. **The INT2 mechanism needs an in-plateau measurement.** The concentration numbers are an
   end-of-training snapshot and `save_total_limit: 1` left no intermediate checkpoints; the
   argument would be much stronger with the coefficient distribution logged *during* the plateau.
5. Wall-clock / memory table for the method vs baselines — not collected.
