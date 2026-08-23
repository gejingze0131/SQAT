#!/usr/bin/env python
"""
Plot the fp16-salient (mixed-precision) sweep: how much of the INT3 quantization gap top-k
protection actually recovers, and what it costs in bits.

Left panel answers "what does the curve look like from k=0 to full precision".
Right panel answers the question behind it: recovery is only meaningful per bit spent, and on
that axis SALT-Q sits above and to the LEFT of the entire curve.

  python scripts/plot_mixedprec_curve.py --out figures/mixedprec_curve.png
"""

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = "results/commonsense_vllm"
TASKS = ["arc_challenge", "arc_easy", "openbookqa", "boolq", "piqa", "siqa",
         "hellaswag", "winogrande"]
TASKS7 = [t for t in TASKS if t != "openbookqa"]

# group_k -> (fp16 share of target weights, effective bits at INT3). o_proj is always group_k=0,
# so the share is computed over q/k/v/gate/up/down only; see scripts/export_mixed_precision_sweep.py.
COST = {0: (0.0, 3.00), 64: (1.21, 3.16), 128: (2.43, 3.32), 256: (4.86, 3.63),
        512: (9.72, 4.26), 1024: (19.43, 5.53), 2048: (38.86, 8.05), None: (100.0, 16.00)}

FLOOR_TAG = "qlora_none_cs170k_int3_ep1-3bit-none-gptq-eval"
FP16_TAG = "qlora_none_cs170k_int3_ep1-3bit-none-merged-eval"
SALTQ_TAG = "saltq_cs170k_int3_g64_ep1_sal5e5-3bit-saltq-deploy-eval"
QALORA_TAG = "qalora_cs170k_int3_ep1-3bit-qalora-dequant-eval"


def load(tag):
    with open(os.path.join(RESULTS, f"{tag}.json")) as f:
        return json.load(f)["results"]


def mean7(r):
    return sum(r[t]["acc"] for t in TASKS7) / len(TASKS7) * 100


def stderr7(r):
    v = [math.sqrt(r[t]["acc"] * (1 - r[t]["acc"]) / r[t]["n"]) * 100 for t in TASKS7]
    return math.sqrt(sum(x * x for x in v)) / len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/mixedprec_curve.png")
    args = ap.parse_args()

    ks = [0, 64, 128, 256, 512, 1024, 2048, None]
    acc, err = {}, {}
    for k in ks:
        tag = FLOOR_TAG if k == 0 else FP16_TAG if k is None else f"mixedprec_int3_g64_k{k}"
        r = load(tag)
        acc[k], err[k] = mean7(r), stderr7(r)

    sq = load(SALTQ_TAG)
    sq_acc, sq_err = mean7(sq), stderr7(sq)
    ql = load(QALORA_TAG)
    ql_acc = mean7(ql)

    floor, ceil = acc[0], acc[None]
    gap = ceil - floor
    rec = lambda a: (a - floor) / gap * 100          # % of the quantization gap recovered

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.axisbelow": True, "figure.facecolor": "white"})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    C_CURVE, C_SALTQ, C_QALORA = "#1f6fb4", "#c0392b", "#7f8c8d"

    # ---------------- left: the sweep, k = 0 .. full ----------------
    x = list(range(len(ks)))
    y = [acc[k] for k in ks]
    e = [err[k] for k in ks]

    axL.axhspan(floor, ceil, color=C_CURVE, alpha=0.05, zorder=0)
    axL.axhline(ceil, color="0.35", lw=1.0, ls=":", zorder=1)
    axL.axhline(floor, color="0.35", lw=1.0, ls=":", zorder=1)

    axL.errorbar(x, y, yerr=e, color=C_CURVE, marker="o", ms=6, lw=2.0,
                 capsize=3, zorder=3, label="fp16 salient + GPTQ rest (no training)")

    axL.axhline(sq_acc, color=C_SALTQ, lw=1.8, ls="--", zorder=2)
    axL.plot([2], [sq_acc], marker="*", ms=20, color=C_SALTQ, zorder=5, linestyle="none")
    axL.annotate(f"SALT-Q  k=128,  a real 3.00-bit model  —  {sq_acc:.2f}",
                 xy=(2, sq_acc), xytext=(2.15, sq_acc + 0.42), color=C_SALTQ,
                 fontsize=10, fontweight="bold")
    axL.axhline(ql_acc, color=C_QALORA, lw=1.4, ls="-.", zorder=2)
    axL.annotate(f"QA-LoRA 3.00 bits — {ql_acc:.2f}", xy=(0.05, ql_acc - 0.36),
                 color=C_QALORA, fontsize=9)

    axL.annotate(f"fp16 merge {ceil:.2f}", xy=(0.05, ceil + 0.12), color="0.3", fontsize=9)
    axL.annotate(f"GPTQ floor {floor:.2f}", xy=(0.05, floor + 0.12), color="0.3", fontsize=9)

    # effective bits under each tick — the curve is unreadable without them
    labels = [("0\n3.00 bit" if k == 0 else "full\n16 bit" if k is None
               else f"{k}\n{COST[k][1]:.2f} bit") for k in ks]
    axL.set_xticks(x)
    axL.set_xticklabels(labels)
    axL.set_xlabel("group_k  (fp16-protected leading columns)   /   effective bit width")
    axL.set_ylabel("Commonsense MEAN(7)  %")
    axL.set_ylim(floor - 0.9, ceil + 0.75)
    axL.set_title("Top-k fp16 protection recovers less than half the gap\n"
                  "before it saturates", fontsize=11, fontweight="bold")

    # right axis: the same curve read as "% of the quantization gap recovered"
    ax2 = axL.twinx()
    ax2.set_ylim((axL.get_ylim()[0] - floor) / gap * 100,
                 (axL.get_ylim()[1] - floor) / gap * 100)
    ax2.set_ylabel("% of the quantization gap recovered", color=C_CURVE)
    ax2.tick_params(axis="y", labelcolor=C_CURVE)
    ax2.grid(False)

    for xi, k in zip(x, ks):
        if k in (0, None):
            continue
        axL.annotate(f"{rec(acc[k]):.0f}%", xy=(xi, acc[k]), xytext=(0, -16),
                     textcoords="offset points", ha="center", fontsize=9, color=C_CURVE)

    axL.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # ---------------- right: recovery per bit spent ----------------
    bx = [COST[k][1] for k in ks]
    by = [rec(acc[k]) for k in ks]
    axR.plot(bx, by, color=C_CURVE, marker="o", ms=6, lw=2.0, zorder=3,
             label="fp16 salient + GPTQ rest (no training)")
    for k, xx, yy in zip(ks, bx, by):
        if k in (0, None):
            continue
        axR.annotate(f"k={k}", xy=(xx, yy), xytext=(4, -12), textcoords="offset points",
                     fontsize=8.5, color=C_CURVE)

    axR.plot([3.00], [rec(sq_acc)], marker="*", ms=22, color=C_SALTQ, linestyle="none",
             zorder=5, label=f"SALT-Q  k=128  ({rec(sq_acc):.0f}% at 3.00 bits)")
    axR.plot([3.00], [rec(ql_acc)], marker="s", ms=7, color=C_QALORA, linestyle="none",
             zorder=4, label=f"QA-LoRA  ({rec(ql_acc):.0f}% at 3.00 bits)")

    axR.annotate("", xy=(3.05, rec(sq_acc)), xytext=(7.9, rec(sq_acc)),
                 arrowprops=dict(arrowstyle="<->", color=C_SALTQ, lw=1.4, ls="--"))
    axR.annotate("the curve needs ~8 bits\nto tie a 3-bit trained model",
                 xy=(5.4, rec(sq_acc) + 1.6), ha="center", color=C_SALTQ,
                 fontsize=9.5, fontweight="bold")

    axR.set_xlabel("effective bit width of the deployed model")
    axR.set_ylabel("% of the quantization gap recovered")
    axR.set_xlim(2.7, 16.6)
    axR.set_ylim(-6, 106)
    axR.set_xticks([3, 4, 5, 6, 8, 10, 12, 14, 16])
    axR.set_title("Read per bit spent, SALT-Q is above and left\n"
                  "of the entire mixed-precision curve", fontsize=11, fontweight="bold")
    axR.legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.suptitle("Does training the salient slice replace keeping it in fp16?   "
                 "Commonsense-170k, Llama-2-7B, INT3 g64, 1 epoch",
                 fontsize=12.5, fontweight="bold", y=0.995)
    fig.text(0.5, 0.008,
             "Mixed-precision arm is PTQ: the QLoRA-merged fp16 checkpoint (82.42), permuted with "
             "SALT-Q's own segmentation, columns [0:k) kept fp16 and the rest GPTQ'd.  "
             "Error bars are +/-1 stderr of MEAN(7).",
             ha="center", fontsize=8.2, color="0.35")
    fig.tight_layout(rect=[0, 0.028, 1, 0.955])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(f"saved -> {args.out}")

    print(f"\n{'k':>6} {'eff bits':>9} {'MEAN(7)':>9} {'recovered':>10}")
    for k in ks:
        lab = "0" if k == 0 else "full" if k is None else str(k)
        print(f"{lab:>6} {COST[k][1]:>9.2f} {acc[k]:>9.2f} {rec(acc[k]):>9.1f}%")
    print(f"{'SALT-Q':>6} {3.00:>9.2f} {sq_acc:>9.2f} {rec(sq_acc):>9.1f}%")
    print(f"{'QA-LoRA':>6} {3.00:>9.2f} {ql_acc:>9.2f} {rec(ql_acc):>9.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
