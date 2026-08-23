# Handoff: the INT2 loss plateau, and why SALT-Q leaves it late

Repo `/home/users/nus/jingzege/projects/SQAT`, branch `saltq/int2-decomposition-and-awq`, head `f7837a3`.

---

## 1. The open question — this is the whole job

At INT2 g32 on Commonsense-170k, **every** method parks on a loss plateau near 0.13 for a large
fraction of the single training epoch. What separates them is only **when they leave it**:

| epoch | 0.10 | 0.30 | 0.45 | 0.60 | 0.75 | 1.00 | final | MEAN(7) |
|---|---|---|---|---|---|---|---|---|
| **QA-LoRA INT2** | 0.1315 | 0.1273 | **0.0836** | 0.0638 | 0.0578 | 0.0541 | 0.0518 | **71.90** |
| **SALT-Q INT2** | 0.1332 | 0.1298 | 0.1280 | 0.1231 | **0.0837** | 0.0680 | 0.0665 | **65.77** |
| SALT-Q, zp_lr ×5 | 0.1332 | 0.1306 | 0.1385 | 0.1258 | — | — | — | running |
| QA-LoRA **INT3** | 0.0703 | 0.0473 | 0.0427 | 0.0373 | 0.0372 | 0.0369 | 0.0355 | 80.92 |

QA-LoRA escapes around epoch **0.45**, SALT-Q around **0.75**. SALT-Q loses ~0.3 of an epoch and
ends **6.13 points behind** (z = −11.0, 0/8 tasks won). At INT3 the plateau does not exist at all.

**Find out why SALT-Q escapes late, and fix it.** That is the deliverable. The 6-point loss to
QA-LoRA is the symptom to explain; do not treat the score as the target directly.

This matters because SALT-Q's parameterization *strictly contains* QA-LoRA's (see §3), so a
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
| `scripts/compare_qalora_saltq_capacity.py --qalora_dir .. --saltq_base .. --saltq_ckpt .. --group_size .. --bits ..` | both methods' coefficient on the shared mean-pooled input, same units |
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

1. **`zp_lr` is not the lever at INT2.** ×5 (8.66e-3) sits on the *same* plateau and escapes
   *later* than the anchor, with more instability (8 grad_norm spikes over 3.0 vs 3, last at epoch
   0.439 vs 0.043). ×10 was killed while queued. Inside the plateau grad_norm is **0.10–0.17**,
   comfortably under `max_grad_norm = 0.3` — **nothing is being clipped, the gradient really is
   that small.** `|Δz| = 0.014 levels` (vs a documented 0.1–0.3 target) is a *symptom* of the
   plateau, not its cause.
2. **`salient_lr` scaling is correct.** `|ΔW_S| = 0.041` grid steps at *both* INT2 and INT3 — the
   grid-step-matched 5e-5 → 1.25e-4 landed exactly on target. The INT3 sweep is sharply unimodal:
   0 → 80.53, 5e-5 → **81.48**, 1e-4 → 80.98, 2e-4 → 80.26, 3.46e-4 → 78.76.
3. **`group_k` is unlikely to be it.** QA-LoRA's INT2 base is `perm_group_k=0` — *zero* protected
   columns — and still reaches 71.90 against SALT-Q's 2.43% protection and 65.77. At INT3 the
   sweep 64/128/256 spans 0.52 over a 4× range, all inside noise.
4. **Two historical runs prove nothing about `zp_lr`** even though earlier notes cited them: `zp3x`
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
| x5 | 8.66e-3 | **0.0670** | **38.28** |

4.8x displacement for 5x lr — the mechanism worked exactly as predicted, and z is demonstrably
neither clamp-limited (0.007% of zero-points sit at the `[0,3]` boundary) nor lr-starved. It moved
INTO the documented `0.1-0.3` band and the model lost 27.5 points, with all 8 tasks below majority
while still emitting well-formed answers ("the correct answer is true" on every boolq item).

That band is MetaMath-derived and this is the **third** time a MetaMath displacement target has
been wrong on Commonsense-170k (`|dW_S| ~0.5` was the first two). Do not treat any of them as a
target here; they are descriptions of a different dataset.

## 6. In flight right now

| job | what | read it how |
|---|---|---|
| `15239837` | SALT-Q INT2, zp_lr ×5 | closes the falsified zp_lr direction; measure `\|Δz\|` |
| `15240708` | **SALT-Q INT2 z-only** (`train_salient: false`) | **the live diagnostic** |

> `15240314` was the first attempt at this run and it **failed after 5 minutes**, so nothing was
> in flight between 00:31 and the relaunch. It passed `--saltq_base_dir` at the anchor's
> `saltq_base_2bit_g32`, and `scripts/train.py`'s reuse guard compares `train_salient` along with
> `(bits, group_size, symmetric)` — a z-only run folds the salient columns into the frozen-code
> pool, so its codes are genuinely different and the guard was right to refuse. The guard's
> message listed only the other three fields, which is why the failure read as inexplicable; it
> now names the field that actually differs. The job no longer passes `--saltq_base_dir` and
> builds its own base under its own `output_root`.

**The z-only run is the current hypothesis test.** The one structural thing SALT-Q carries through
the plateau that QA-LoRA does not is the salient QAT tier: 157.3M real weights trained through a
**2-bit straight-through estimator** on a grid so coarse that `|ΔW_S| = 0.041` steps **can never
flip a code** (0.5 would be needed). A biased STE gradient from a tier that cannot act is a
plausible thing to hold the model on the plateau. `train_salient: false` removes exactly that tier
and nothing else, and makes SALT-Q the full-rank version of QA-LoRA.

Read rule, fixed before the run — **read the escape epoch from the loss curve, not the score**
(z-only cost 0.95 points at INT3, so a lower score is expected and settles nothing):

- escape near **0.45**, like QA-LoRA → the salient tier delays the escape; the fix is in how the
  salient slice is trained, and no learning rate will do it.
- escape still near **0.75** → the salient tier is exonerated; the delay lives in the frozen-code
  / (s,z) path, and the base is the next thing to look at.

Extract the curve with:

```bash
grep -oE "\{'loss': '[0-9.eE+-]+', 'grad_norm': '[0-9.eE+-]+',[^}]*'epoch': '[0-9.]+'\}" \
  logs/commonsense_170k/<run>.log
```

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
