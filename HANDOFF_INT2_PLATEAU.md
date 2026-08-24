# Handoff: the INT2 loss plateau, and why SALT-Q leaves it late

Repo `/home/users/nus/jingzege/projects/SQAT`, branch `saltq/int2-decomposition-and-awq`, head `e48ea93`.

---

## 1. The open question — this is the whole job

At INT2 g32 on Commonsense-170k, **every** method parks on a loss plateau near 0.13 for a large
fraction of the single training epoch. What separates them is only **when they leave it**:

| epoch | 0.10 | 0.30 | 0.45 | 0.60 | 0.75 | 1.00 | **escape** | MEAN(7) |
|---|---|---|---|---|---|---|---|---|
| **QA-LoRA INT2** | 0.1315 | 0.1273 | **0.0836** | 0.0638 | 0.0578 | 0.0541 | **0.347** | **71.90** |
| **SALT-Q INT2** | 0.1332 | 0.1298 | 0.1280 | 0.1231 | **0.0837** | 0.0680 | **0.672** | **65.77** |
| SALT-Q z-only | 0.1321 | 0.1294 | 0.1266 | 0.1236 | 0.1215 | 0.1151 | **0.938** | 48.54 |
| SALT-Q zp_lr x5 | 0.1332 | 0.1306 | 0.1385 | 0.1258 | 0.1281 | 0.1287 | **never** | 38.28 |
| QLoRA INT2 floor (PTQ) | — | — | — | — | — | — | — | 50.35 |
| QA-LoRA **INT3** | 0.0703 | 0.0473 | 0.0427 | 0.0373 | 0.0372 | 0.0369 | <0.10 | 80.92 |

(escape = first logged step under loss 0.110. All are 1 epoch, INT2 g32, group_k=128 unless noted.)

**The score is monotone in the escape epoch across every run.** QA-LoRA escapes at 0.347 and
scores 71.90; SALT-Q at 0.672 and 65.77; z-only at 0.938 and 48.54; the x5 run never escapes and
collapses to 38.28 with all eight tasks degenerate. At INT3 no plateau exists at all.

**Explain the escape ordering and you have explained the 6.13-point loss.** That is the
deliverable. Do not chase the score directly — every run that tried to raise it by turning a
learning rate up made the escape later, not earlier.

This matters because SALT-Q's parameterization *strictly contains* QA-LoRA's (see §2), so a
6-point loss cannot be an expressiveness result. It is an optimization result.

---

## 2. What SALT-Q is

Per weight matrix, after a saliency-ordering permutation of the input channels:

- columns `[0 : group_k)` — **salient**: real fp weights, trainable, seen through an LSQ
  fakequant at `q_bits`. Deployed quantized like everything else, so **group_k costs zero
  deployment bits**.
- columns `[group_k : in)` — **non-salient**: GPTQ integer codes, **frozen forever**. Only the
  per-(output, group) scale `s` and zero-point `z` are trainable.
- `o_proj` always has `group_k = 0` (structural: a salient slice there breaks the grouping).

Three parameter tiers, three learning rates, each in **a different unit**:

| tier | param | unit | INT3 anchor | INT2 anchor |
|---|---|---|---|---|
| salient weights | `weight_salient` | weight units | `salient_lr` 5.0e-5 | 1.25e-4 |
| salient LSQ scale | `lsq_w_scale` | weight units | `scales_lr` 1.73e-5 | 1.73e-5 |
| **zero-points** | `saltq_z` | **quantization LEVELS** | `zp_lr` 1.73e-3 | 1.73e-3 |

`z` is clamped to `[0, 2^b − 1]` → `[0,7]` at INT3 but only **`[0,3]` at INT2**
(`src/qat_saltq.py:513`, `:628`). The grid step `s(b) = range/(2^b − 1)` is 2.33× coarser at INT2,
so any rate expressed in weight units means something different at each bit width.

Supporting machinery: a segment-permuted residual stream with a `BoundaryGatherHook` per segment
boundary, folded into weights at export (`fold_boundary_gathers_into_weights`).

**Baselines.** *QA-LoRA*: LoRA r=64 α=16 (scaling 0.25), lr 5e-3, folded into the zero-points of
a GPTQ base built with `perm_group_k=0` — **no protected columns at all**. *QLoRA floor*: LoRA on
an NF4 base, merged, then GPTQ'd post-hoc.

**Shared function space** (this is the key framing, and `scripts/compare_qalora_saltq_capacity.py`
measures it): both methods add a per-(output, group) coefficient on a pooled input.

```
QA-LoRA   y += scaling * (B @ A) @ mean_g(x)      (B@A) is [out, G], rank <= 64
SALT-Q    y -= (z * s)          @ sum_g(x)        (z*s) is [out, G], FULL RANK
```

QA-LoRA pools with a **mean**, SALT-Q with a **sum**, so a SALT-Q coefficient converts as
`Δz · s · group_size`. SALT-Q's reachable set contains QA-LoRA's, *plus* it trains the salient
weights. Expressiveness is not the explanation for anything here.

---

## 3. Project structure

```
configs/*.yaml          one file per experimental cell; headers carry the derivation + prior results
runs/<method>/          run_<method>_<dataset>.sh  ->  execs _pipeline.sh with --dataset/--config fixed
runs/eval_vllm.sh       eval-only; hops conda envs, folds the residual permutation, writes results/
runs/lib/common.sh      cd_repo_root, assert_config_matches_dataset, config_output_dir
jobs/*.pbs              one per submitted run; the header is the experiment's rationale + read rules
scripts/train.py        the trainer entry point (accelerate launch)
scripts/*.py            diagnostics, exports, collectors (see below)
src/                    9.8k LOC: qat_saltq.py, gptq.py, permute_common.py, qalora.py, export.py, ...
results/commonsense_vllm/<tag>.json     per-task acc; .jsonl has raw generations   (GITIGNORED)
outputs/                checkpoints, bases, export dirs                            (GITIGNORED)
logs/commonsense_170k/  training + eval logs
```

**Environments.** `conda activate saltq` for training; `vllm-eval` (on `/scratch`, vLLM 0.11.0 +
torch 2.8.0 cu128) for generation — `runs/eval_vllm.sh` hops between them itself.
**Scheduler.** PBS: `qsub jobs/<name>.pbs`, `qstat -u $USER`, `qdel <id>`. 4 GPUs per job, and a
full train+export+eval cycle is **~3–4 h** (train ~1.5 h, GPTQ base ~3 min, eval ~1 min of vLLM
plus model load).

### Running things

```bash
bash runs/saltq/run_saltq_commonsense.sh \
    --config configs/<cell>.yaml --bits 2 --output_root outputs/<cell> \
    --permuted_base_dir outputs/saltq_cs170k_int3_g64_ep1/permuted_fp16_base \
    --saltq_base_dir    outputs/saltq_cs170k_int2_g32_ep1/saltq_base_2bit_g32
bash runs/qalora/run_qalora_commonsense.sh --config configs/<cell>.yaml --bits 2 --output_dir ...
bash runs/eval_vllm.sh --model_path <dir> --dataset commonsense --gpus 0,1,2,3 --tag <tag>
```

The **permuted fp16 base** is bit-independent (the segmentation DP never sees bits or group_size)
and can be reused across bit widths — `scripts/train.py`'s permutation fingerprint refuses the
reuse if that is ever wrong. The **frozen GPTQ base** (`saltq_base_{bits}bit_g{gs}`) must be
rebuilt whenever bits, group_size **or group_k** changes. Changing `group_k` also changes the
segmentation, so it needs a fresh permuted base too (see `jobs/cs_saltq_int3_ep1_k256.pbs`).

### Reading results

```python
import json, math
T  = ["arc_challenge","arc_easy","openbookqa","boolq","piqa","siqa","hellaswag","winogrande"]
T7 = [t for t in T if t != "openbookqa"]          # openbookqa: 500 examples, stderr 2.4 — EXCLUDE
def load(tag): return json.load(open(f"results/commonsense_vllm/{tag}.json"))["results"]
def se(v):  p, n = v["acc"], v["n"]; return math.sqrt(p*(1-p)/n)*100
def mean(r, ts): return sum(r[t]["acc"] for t in ts)/len(ts)*100
def mse(r, ts):  return math.sqrt(sum(se(r[t])**2 for t in ts))/len(ts)
```

Each task dict also carries `dominant_output_share`, `beats_majority`, `majority_class_acc`.
**Always check them at INT2** — a model that has collapsed to one answer per task can still post a
plausible-looking mean.

### Diagnostics that already exist

| script | what it gives |
|---|---|
| `scripts/measure_saltq_displacement.py --base <base> --ckpt <final> --salient_lr .. --scales_lr .. --zp_lr ..` | how far each tier moved, in its own unit: `\|ΔW_S\|` in grid steps, `\|Δz\|` in levels, `\|Δs\|` relative |
| `scripts/compare_qalora_saltq_capacity.py --qalora_dir .. --saltq_base .. --saltq_ckpt .. --group_size .. --bits ..` | both methods' coefficient on the shared mean-pooled input, same units. Handles both checkpoint layouts: a z-only run's `saltq_z` spans every group while the base's `z_n` spans only the non-salient ones (344 vs 340 for down_proj at g32/k=128), so the base is rebuilt as `[z_s \| z_n]` |
| `scripts/test_saltq_base_provenance.py` | 5 checks that stale frozen codes cannot reach training |
| `scripts/test_runs_wiring.sh` | every entry script's documented flags are actually parsed |
| `scripts/plot_mixedprec_curve.py` | the fp16-salient sweep figure |

---

## 4. Everything measured so far

**INT3 g64, 1 epoch, MEAN(7) ± stderr.** Pairwise resolution is **±0.48**, so anything under ~0.95
is unreadable at one seed.

| | MEAN(7) |
|---|---|
| fp16 merge (upper bound) | 82.42 ± 0.33 |
| **SALT-Q k=128, salient_lr 5e-5** | **81.48 ± 0.34** |
| SALT-Q k=256 / k=64 | 81.61 / 81.09 |
| QA-LoRA | 80.92 ± 0.34 |
| SALT-Q z-only (`train_salient: false`) | 80.53 |
| PSQAT (LoRA + frozen GPTQ) | 79.29 |
| QLoRA INT3 GPTQ floor | 78.38 ± 0.36 |

**INT2 g32, 1 epoch.**

| | MEAN(7) | note |
|---|---|---|
| QA-LoRA | **71.90 ± 0.38** | no collapsed tasks |
| SALT-Q k=128 | **65.77 ± 0.40** | boolq collapsed, below majority |
| QLoRA INT2 GPTQ floor | 50.35 ± 0.42 | boolq + hellaswag below majority — a destroyed model |

**The 2×2 decomposition at INT3** (from the frozen-GPTQ floor):

| addition | Δ | z | verdict |
|---|---|---|---|
| trainable (s,z) alone | +2.15 | 4.34 | significant |
| LoRA alone (PSQAT) | +0.91 | 1.82 | noise |
| QAT salient on top of z | +0.95 | 1.98 | significant |
| SALT-Q vs PSQAT | +2.19 | 4.51 | significant, 8/8 tasks, sign p=0.0078 |

Mechanism: what matters is **whether the adaptation survives into the deployed quantization
parameters**, not rank and not the LoRA form. Low-rank vs full-rank z measured at −0.39 (noise).

**Mixed-precision sweep at INT3** (training-free: QLoRA merged checkpoint, SALT-Q's own
permutation replayed, columns `[0:k)` kept fp16, rest GPTQ'd — `scripts/export_mixed_precision_sweep.py`,
`keep_salient_fp16=True`). Gap to close is 4.04 points:

| k | fp16 share | eff bits | MEAN(7) | gap recovered |
|---|---|---|---|---|
| 0 | 0% | 3.00 | 78.38 | 0% |
| 64 | 1.21% | 3.16 | 79.35 | 24.0% |
| 128 | 2.43% | 3.32 | 80.30 | 47.4% |
| 256 | 4.86% | 3.63 | 79.97 | 39.4% |
| 512 | 9.72% | 4.26 | 80.33 | 48.2% |
| 1024 | 19.43% | 5.53 | 80.56 | 53.9% |
| 2048 | 38.86% | 8.05 | 81.18 | 69.3% |
| full | 100% | 16.00 | 82.42 | 100% |
| **SALT-Q k=128** | — | **3.00** | **81.48** | **76.7%** |

The curve **never reaches SALT-Q** — training is worth ~2.5–5 bits. It is also flat from k=128 to
k=1024, which is the group_k negative seen from the other side: outlier energy is concentrated.
Caveat to keep stating: that arm is **PTQ**, so it is QAT vs PTQ-mixed-precision, not QAT vs QAT.

**Other settled facts.** `group_by_length: true` collapses commonsense (37.03 vs 79.04) because
task lengths barely overlap and batches go task-pure — must stay `false`. Segmentation is not a
lever (autoseg 79.04 vs manual `[2,30]` 78.87). 3 epochs at INT3 gave no clear gain.

---

## 5. Ruled out — do not re-run these

1. **`zp_lr` is not the lever at INT2 — it is actively harmful.** ×5 (8.66e-3) never leaves the
   plateau at all and collapses to 38.28 with all eight tasks degenerate. ×10 was killed while
   queued. Inside the plateau grad_norm is **0.10–0.17**, comfortably under `max_grad_norm = 0.3`
   — **nothing is being clipped, the gradient really is that small.** `|Δz| = 0.0143 levels` is a
   *symptom* of the plateau, not its cause; ×5 moved it to 0.0707, INTO the documented 0.1–0.3
   band's neighbourhood, and the model got 27.5 points worse. See §5b: the deficit is not
   magnitude.
2. **The salient QAT tier is not the problem — removing it costs 17.23 points** (§6). The
   straight-through-estimator hypothesis is falsified; do not re-test it.
3. **`salient_lr` scaling is correct.** `|ΔW_S| = 0.041` grid steps at *both* INT2 and INT3 — the
   grid-step-matched 5e-5 → 1.25e-4 landed exactly on target. The INT3 sweep is sharply unimodal:
   0 → 80.53, 5e-5 → **81.48**, 1e-4 → 80.98, 2e-4 → 80.26, 3.46e-4 → 78.76.
4. **`group_k` is unlikely to be it.** QA-LoRA's INT2 base is `perm_group_k=0` — *zero* protected
   columns — and still reaches 71.90 against SALT-Q's 2.43% protection and 65.77. At INT3 the
   sweep 64/128/256 spans 0.52 over a 4× range, all inside noise.
5. **Two historical runs prove nothing about `zp_lr`** even though earlier notes cited them: `zp3x`
   paired zp_lr 5.19e-3 with salient_lr 3.46e-4, and the `lr` run paired zp_lr 1.73e-2 with
   salient_lr 6.34e-4, scales_lr ×1.88 **and** `train_layernorms: true`. Both failures are
   attributable to salient_lr (3.46e-4 alone costs 2.7 points). **There has never been a clean
   zp_lr experiment on this dataset.**

---

### The x5 zp_lr point closed the lr question by confirming its own prediction

The job stated the falsifiable version in advance: Adam displacement is ~linear in lr, so x5
should put `|dz|` near 0.07. Measured on the finished checkpoint:

| | zp_lr | `\|dz\|` p50 (levels) | MEAN(7) |
|---|---|---|---|
| anchor | 1.73e-3 | 0.0143 | 65.77 |
| x5 | 8.66e-3 | **0.0707** | **38.28** |

4.94x displacement for 5x lr (all 224 projections) — the mechanism worked exactly as predicted, and z is demonstrably
neither clamp-limited (0.00% of zero-points sit at the `[0,3]` boundary) nor lr-starved. It moved
INTO the documented `0.1-0.3` band and the model lost 27.5 points, with all 8 tasks below majority
while still emitting well-formed answers ("the correct answer is true" on every boolq item).

That band is MetaMath-derived and this is the **third** time a MetaMath displacement target has
been wrong on Commonsense-170k (`|dW_S| ~0.5` was the first two). Do not treat any of them as a
target here; they are descriptions of a different dataset.

## 5b. What actually separates them: SHAPE, not size

`scripts/compare_qalora_saltq_capacity.py` puts both methods' update in one unit -- the
coefficient each applies to the mean-pooled input (`(B@A)*scaling` for QA-LoRA,
`-dz*s*group_size` for SALT-Q). End of training, INT2 g32:

| run | p50 | p90 | p99 | mean | **p99/p50** | \|dz\| levels | escape | MEAN(7) |
|---|---|---|---|---|---|---|---|---|
| QA-LoRA | 9.950e-3 | 5.665e-2 | 1.527e-1 | 2.187e-2 | **15.3** | — | 0.347 | 71.90 |
| SALT-Q anchor | 1.125e-2 | 3.285e-2 | 6.370e-2 | 1.502e-2 | **5.7** | 0.0143 | 0.672 | 65.77 |
| SALT-Q z-only | 6.600e-3 | 1.885e-2 | 3.790e-2 | 8.792e-3 | **5.7** | 0.0084 | 0.938 | 48.54 |
| SALT-Q zp_lr x5 | 5.605e-2 | 1.537e-1 | 2.763e-1 | 7.156e-2 | **4.9** | 0.0707 | never | 38.28 |
| (INT3 QA-LoRA | 1.165e-2 | 6.065e-2 | 1.628e-1 | 2.387e-2 | 14.0 | — | <0.10 | 80.92) |
| (INT3 SALT-Q | 1.215e-2 | 4.385e-2 | 9.095e-2 | 1.858e-2 | 7.5 | 0.0158 | — | 81.48) |

Four things are now settled by measurement rather than argued.

**1. The zero-point clamp is not the limiter.** 0.00% of groups sit at either boundary at both bit
widths (`z_init` p50 is 2.000 of `[0,3]` at INT2, 3.000 of `[0,7]` at INT3). That hypothesis is
dead; `zp_lr` does not run into a wall.

**2. `|dz|` is exactly lr-linear, and the registered prediction held.** 5x `zp_lr` moved `|dz|`
from 0.0143 to 0.0707 levels -- **4.94x**, against a predicted ~5x. z is fully lr-controllable.

**3. The deficit is therefore NOT magnitude.** The x5 run's mean coefficient is **3.3x LARGER than
QA-LoRA's**, and it never escaped and scored 38.28 with every task degenerate. Across the three
SALT-Q runs the magnitude spans **8.1x** while the score falls monotonically. More update is worse.

**4. What does not change is the SHAPE.** `p99/p50` is 4.9-5.7 for every SALT-Q variant, across
that entire 8.1x magnitude range, while QA-LoRA sits at 15.3 (and 14.0 at INT3 -- QA-LoRA holds
its shape across bit widths too). **SALT-Q's update is uniform and no learning rate makes it less
so.** Rank <= 64 on an `[out, G]` matrix is a sum of 64 outer products and *must* pile onto a few
directions; full-rank `z` under Adam gets per-parameter normalisation, so every coefficient takes
about the same step and nothing can concentrate.

**Working hypothesis: leaving the plateau needs a few large targeted corrections, and a full-rank
per-coefficient parameterization under Adam can only produce uniformly small ones.**

It accounts for every run: raising `zp_lr` scales all coefficients equally and adds no
concentration, which is why x5 was catastrophic rather than merely unhelpful; and at INT3 the
frozen codes are good enough that uniform small corrections suffice, which is exactly where
full-rank `z` becomes an advantage and SALT-Q wins (81.48 vs 80.92).

### The gap in it, stated plainly

`p99/p50` is **identical (5.7) for the anchor and for z-only**, yet they escape 0.27 epochs apart
and differ by 17.23 points. So the per-group coefficient's shape does not explain that pair. What
separates them is the salient tier, and that tier's update is **a different kind of object** --
per-element on 128 real-weight columns, which no per-group coefficient can express and which this
measurement does not capture at all. There are plausibly two concentration mechanisms in play and
SALT-Q has the weaker version of each. A measurement that covers the salient tier in the same
units would settle it.

Also: these are **end-of-training snapshots**, not measurements taken on the plateau, and
`save_total_limit: 1` leaves no intermediate checkpoints. Logging the coefficient distribution
*during* the plateau is the obvious next instrument, and nothing here is causal until that exists.

---

## 6. The salient tier is EXONERATED -- and it was helping

The z-only run (`train_salient: false`) was submitted to test whether the salient QAT tier holds
the model on the plateau: 157.3M real weights trained through a 2-bit straight-through estimator
on a grid so coarse that `|dW_S| = 0.041` steps can never flip a code (0.5 is needed). A biased
gradient from a tier that cannot act seemed a plausible culprit.

**The opposite is true.** Removing it pushed the escape from 0.672 to **0.938** and cost **17.23
points** (65.77 -> 48.54, boolq and winogrande both degenerate). The salient tier is not what
pins SALT-Q to the plateau; it is the main thing dragging it off. The STE hypothesis is dead.

That also reorders the whole picture by how *concentrated* an update each configuration can make:

| | can it concentrate? | escape | MEAN(7) |
|---|---|---|---|
| QA-LoRA | low rank — **forced** to | 0.347 | 71.90 |
| SALT-Q anchor | uniform z **+ 128 real-weight columns** | 0.672 | 65.77 |
| SALT-Q z-only | uniform z only — **maximally flat** | 0.938 | 48.54 |
| SALT-Q zp_lr x5 | uniform z, scaled up — noise, no targeting | never | 38.28 |

Perfectly monotone, and it is the ordering §5b predicts.

### One piece of history worth keeping

`15240314` was the first attempt at the z-only run and it **failed after 5 minutes**; `15240708`
is the one that produced the numbers above. The first passed `--saltq_base_dir` pointing at the
anchor's `saltq_base_2bit_g32`, and `scripts/train.py`'s reuse guard compares `train_salient`
along with `(bits, group_size, symmetric)` — a z-only run folds the salient columns into the
frozen-code pool, so its codes really are different and the guard was right to refuse. The guard's
message listed only the other three fields, which is why the refusal read as inexplicable; it now
names the field that actually differs. **A z-only cell must build its own base under its own
`output_root`** — do not hand it a `train_salient: true` base.

### Nothing is in flight

All INT2 jobs have finished. `15239838` (zp_lr x10) was killed while queued once x5 falsified that
direction. The open question in §1 is unanswered.

### The two instruments worth building next

1. **Log the coefficient distribution during training**, not just at the end. `save_total_limit: 1`
   currently destroys the evidence. Without this every claim in §5b stays correlational.
2. **Put the salient tier into the same units as the per-group coefficient**, so the anchor-vs-
   z-only gap (§5b, "the gap in it") can be attributed instead of assumed.

---

## 7. Methodological rules that have already cost time

- **Change one variable per run.** Two failed runs above are uninterpretable because they moved
  four knobs at once, and one of them was later shown to be decisive on its own.
- **Displacement targets from MetaMath do not transfer.** `|ΔW_S| ~0.5` steps and `|Δz| 0.1–0.3`
  levels have both been wrong on Commonsense-170k. The best INT3 score came from 0.041 steps —
  twelve times below the "target" — while an in-band 0.273 was the worst run of its sweep.
  Re-derive per dataset; never act on the band alone.
- **Read MEAN(7), and read ordering/sign tests alongside pairwise z.** One seed resolves ±0.48.
- **Check `dominant_output_share` / `beats_majority` before believing any INT2 number.**
- **Never pipe a long-running python diagnostic through `tail`.** `compare_qalora_saltq_capacity.py`
  was OOM-killed for weeks' worth of invocations and the pipeline still exited 0 with the buffered
  stdout lost, so it looked like it had succeeded and printed nothing. Fixed in `f7837a3`.
- Write the rationale **and the read rule** into the `.pbs` header before submitting, so the
  interpretation is fixed before the number arrives.
- **Read the trajectory, not just the final score.** Every INT2 conclusion in this document came
  from aligning loss curves by epoch. The scores alone say "SALT-Q is worse"; the curves say
  "SALT-Q leaves the plateau 0.3 epochs later", which is a different and answerable question.
- **Separate magnitude from shape before blaming a learning rate.** SALT-Q's update magnitude
  spans 8.1x across three runs while its `p99/p50` never leaves 4.9-5.7. Two of those runs were
  submitted on the theory that magnitude was the deficit; both made things worse.
- **State the falsifiable prediction in the `.pbs` header.** The x5 zp_lr job predicted
  `|dz| ~ 0.07` and measured 0.0707, which is what let a bad score be read as "the mechanism works
  and the hypothesis is wrong" instead of "something broke".
