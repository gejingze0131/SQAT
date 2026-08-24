#!/usr/bin/env python
"""
fp16-salient (mixed precision) vs SALT-Q, both swept over group_k, Commonsense INT3 g64.

The question: does training the salient slice replace keeping it in fp16, and how much of the
quantization gap does top-k protection actually contribute?

Both arms sweep group_k over the same segmentation, the same salient channels and the same GPTQ
code path, so every point on the left panel is a MATCHED pair differing only in what made the
salient slice good. SALT-Q holds 3.00 effective bits at every k; the fp16 arm pays bits to move
right.

  --metric mean8   the 8-task mean the result tables report (default)
  --metric mean7   drops openbookqa, whose 500 test examples carry a ~1.7-point stderr

Both are shipped because the choice moves the headline: at k=256 SALT-Q recovers 83.5% of the gap
on MEAN(8) but 79.9% on MEAN(7), and openbookqa alone swings 82.60 -> 84.60 between k=128 and
k=256. The conclusion is the same either way, which is the point of being able to render both.

Colors are slots 1-3 of the data-viz reference palette (blue/orange/aqua), used unmodified; that
three-slot set is the documented all-pairs-validated subset in both modes.

  python scripts/plot_mixedprec_curve.py --metric mean8 --out figures/mixedprec_curve.png
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "results/commonsense_vllm"
TASKS8 = ["arc_challenge", "arc_easy", "openbookqa", "boolq", "piqa", "siqa",
          "hellaswag", "winogrande"]
TASKS7 = [t for t in TASKS8 if t != "openbookqa"]

# group_k -> effective bit width of the DEPLOYED fp16-salient model at INT3. o_proj is always
# group_k=0, so the fp16 share counts q/k/v/gate/up/down only. See export_mixed_precision_sweep.py.
MP_BITS = {0: 3.00, 64: 3.16, 128: 3.32, 256: 3.63, 512: 4.26, 1024: 5.53, 2048: 8.05, None: 16.00}

FLOOR = "qlora_none_cs170k_int3_ep1-3bit-none-gptq-eval"
CEIL = "qlora_none_cs170k_int3_ep1-3bit-none-merged-eval"
QALORA = "qalora_cs170k_int3_ep1-3bit-qalora-dequant-eval"
SALTQ = {0: "saltq_cs170k_int3_g64_ep1_zonly-3bit-saltq-deploy-eval",
         64: "saltq_cs170k_int3_ep1_k64-3bit-saltq-deploy-eval",
         128: "saltq_cs170k_int3_g64_ep1_sal5e5-3bit-saltq-deploy-eval",
         256: "saltq_cs170k_int3_ep1_k256-3bit-saltq-deploy-eval"}

# data-viz reference palette, categorical slots 1-3 (light mode)
C_SQ, C_MP, C_QA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"


def load(tag):
    with open(os.path.join(RESULTS, f"{tag}.json")) as f:
        return json.load(f)["results"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["mean8", "mean7"], default="mean8")
    ap.add_argument("--out", default="figures/mixedprec_curve.png")
    args = ap.parse_args()

    tasks = TASKS8 if args.metric == "mean8" else TASKS7
    other = TASKS7 if args.metric == "mean8" else TASKS8
    mlabel = f"MEAN({len(tasks)})"

    def agg(r, ts=None):
        ts = ts or tasks
        return sum(r[t]["acc"] for t in ts) / len(ts) * 100

    def err(r, ts=None):
        ts = ts or tasks
        v = [math.sqrt(r[t]["acc"] * (1 - r[t]["acc"]) / r[t]["n"]) * 100 for t in ts]
        return math.sqrt(sum(x * x for x in v)) / len(v)

    mp_ks = [0, 64, 128, 256, 512, 1024, 2048, None]
    mp = {}
    for k in mp_ks:
        tag = FLOOR if k == 0 else CEIL if k is None else f"mixedprec_int3_g64_k{k}"
        r = load(tag)
        mp[k] = (agg(r), err(r))
    sq, sq_other = {}, {}
    for k, t in SALTQ.items():
        r = load(t)
        sq[k], sq_other[k] = (agg(r), err(r)), agg(r, other)
    qa = agg(load(QALORA))

    floor, ceil_ = mp[0][0], mp[None][0]
    gap = ceil_ - floor
    rec = lambda a: (a - floor) / gap * 100
    # the same quantity under the other task set, for the caption's sensitivity line
    fo, co = agg(load(FLOOR), other), agg(load(CEIL), other)
    rec_o = lambda a: (a - fo) / (co - fo) * 100

    plt.rcParams.update({
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#d9d8d2", "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
    })
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.9))

    # ================= left: the metric vs group_k, matched pairs =================
    xs = {k: i for i, k in enumerate(mp_ks)}

    axL.axhspan(floor, ceil_, color=INK3, alpha=0.07, zorder=0)
    for yv, lab in ((ceil_, f"fp16 merge  {ceil_:.2f}   (gap = {gap:.2f} pts)"),
                    (floor, f"GPTQ INT3 floor  {floor:.2f}")):
        axL.axhline(yv, color=INK3, lw=1.0, ls=(0, (2, 2)), zorder=1)
        axL.annotate(lab, xy=(0.04, yv + 0.10), color=INK2, fontsize=8.8)

    axL.errorbar([xs[k] for k in mp_ks], [mp[k][0] for k in mp_ks],
                 yerr=[mp[k][1] for k in mp_ks], color=C_MP, marker="o", ms=8, lw=2.0,
                 capsize=3, zorder=3, label="fp16 salient + GPTQ rest  (PTQ, pays bits)")
    axL.errorbar([xs[k] for k in sq], [sq[k][0] for k in sq],
                 yerr=[sq[k][1] for k in sq], color=C_SQ, marker="o", ms=8, lw=2.2,
                 capsize=3, zorder=4, label="SALT-Q trained salient  (always 3.00 bits)")
    axL.plot([xs[0]], [qa], marker="s", ms=9, color=C_QA, ls="none", zorder=4,
             label="QA-LoRA  (3.00 bits)")

    # effective bits on the fp16 arm only -- SALT-Q is 3.00 everywhere, so a shared tick label
    # would be wrong for half the chart.
    for k in mp_ks:
        if k in (0, None):
            continue
        axL.annotate(f"{MP_BITS[k]:.2f} bit", xy=(xs[k], mp[k][0]), xytext=(0, -24),
                     textcoords="offset points", ha="center", fontsize=8.4, color=C_MP,
                     bbox=dict(fc="#fcfcfb", ec="none", pad=1.1))
    axL.annotate("3.00 bit\nthroughout", xy=(xs[256], sq[256][0]), xytext=(14, -6),
                 textcoords="offset points", fontsize=9, color=C_SQ, fontweight="bold")
    axL.annotate(f"QA-LoRA {qa:.2f}", xy=(xs[0], qa), xytext=(34, 26),
                 textcoords="offset points", fontsize=8.8, color=C_QA,
                 arrowprops=dict(arrowstyle="-", color=C_QA, lw=0.9, shrinkA=1, shrinkB=3))
    axL.annotate(f"z-only (k=0)  {sq[0][0]:.2f}", xy=(xs[0], sq[0][0]), xytext=(11, -14),
                 textcoords="offset points", fontsize=8.6, color=C_SQ)
    axL.annotate(f"{sq[256][0]:.2f}", xy=(xs[256], sq[256][0]), xytext=(0, 13),
                 textcoords="offset points", ha="center", fontsize=8.8, color=C_SQ)

    axL.set_xticks(list(xs.values()))
    axL.set_xticklabels(["0" if k == 0 else "full" if k is None else str(k) for k in mp_ks])
    axL.set_xlabel("group_k   —   leading columns given the salient treatment")
    axL.set_ylabel(f"Commonsense-170k  {mlabel}   %")
    axL.set_ylim(floor - 1.15, ceil_ + 0.60)
    axL.set_title("At every matched group_k, training the salient slice beats\n"
                  "keeping it in fp16 — and costs no extra bits",
                  fontsize=11.5, fontweight="bold", color=INK)
    h, l = axL.get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: ("SALT-Q" not in l[i], "QA-LoRA" in l[i]))
    axL.legend([h[i] for i in order], [l[i] for i in order],
               loc="lower right", fontsize=9, framealpha=0.96, edgecolor="#d9d8d2")

    # ================= right: recovery vs what it costs =================
    axR.axhline(100, color=INK3, lw=1.0, ls=(0, (2, 2)), zorder=1)
    axR.annotate("fp16 merge = 100% of the gap", xy=(6.4, 101.6), color=INK2, fontsize=8.8)

    axR.plot([MP_BITS[k] for k in mp_ks], [rec(mp[k][0]) for k in mp_ks],
             color=C_MP, marker="o", ms=8, lw=2.0, zorder=3,
             label="fp16 salient + GPTQ rest")
    for k in mp_ks:
        if k in (0, None):
            continue
        axR.annotate(f"k={k}", xy=(MP_BITS[k], rec(mp[k][0])), xytext=(8, -13),
                     textcoords="offset points", fontsize=8.4, color=C_MP)

    sq_ks = sorted(sq)
    axR.plot([3.00] * len(sq_ks), [rec(sq[k][0]) for k in sq_ks],
             color=C_SQ, marker="o", ms=8, lw=2.2, zorder=4,
             label="SALT-Q  (k = 0, 64, 128, 256)")
    axR.annotate(f"k={sq_ks[0]}", xy=(3.00, rec(sq[sq_ks[0]][0])), xytext=(-8, -3),
                 textcoords="offset points", fontsize=8.6, color=C_SQ, ha="right")
    axR.annotate(f"k={sq_ks[-1]}", xy=(3.00, rec(sq[sq_ks[-1]][0])), xytext=(7, 8),
                 textcoords="offset points", fontsize=8.6, color=C_SQ)
    axR.plot([3.00], [rec(qa)], marker="s", ms=9, color=C_QA, ls="none", zorder=4,
             label="QA-LoRA")
    axR.annotate("QA-LoRA", xy=(3.00, rec(qa)), xytext=(10, -10),
                 textcoords="offset points", fontsize=8.6, color=C_QA)

    # ONE annotation: the strongest claim the panel supports. Two arrows in this space collided
    # with each other and with the curve, so the z-only equivalence lives in the caption.
    axR.annotate("", xy=(3.03, rec(sq[256][0])), xytext=(8.05, rec(sq[256][0])),
                 arrowprops=dict(arrowstyle="<->", color=C_SQ, lw=1.5, shrinkA=0, shrinkB=0))
    axR.annotate(f"SALT-Q k=256 recovers {rec(sq[256][0]):.0f}% at 3.00 bits, and is\n"
                 f"only TIED once the fp16 arm reaches 8.05",
                 xy=(4.9, min(91.5, rec(sq[256][0]) + 9.0)), ha="center", fontsize=9.6,
                 color=C_SQ, fontweight="bold")

    axR.set_xscale("log", base=2)
    axR.set_xlim(2.82, 18.5)
    axR.set_ylim(-7, 112)
    axR.set_xticks([3, 3.5, 4, 5, 6, 8, 11, 16])
    axR.set_xticklabels(["3", "3.5", "4", "5", "6", "8", "11", "16"])
    axR.minorticks_off()
    axR.set_xlabel("effective bit width of the deployed model")
    axR.set_ylabel(f"% of the {gap:.2f}-point quantization gap recovered")
    axR.set_title("Read per bit spent: SALT-Q climbs straight up at 3 bits,\n"
                  "the fp16 arm has to walk right",
                  fontsize=11.5, fontweight="bold", color=INK)
    hR, lR = axR.get_legend_handles_labels()
    oR = sorted(range(len(lR)), key=lambda i: ("SALT-Q" not in lR[i], "QA-LoRA" in lR[i]))
    axR.legend([hR[i] for i in oR], [lR[i] for i in oR],
               loc="lower right", fontsize=9, framealpha=0.96, edgecolor="#d9d8d2")

    fig.suptitle("Does training the salient slice replace keeping it in fp16?    "
                 "Llama-2-7B, Commonsense-170k, INT3 g64, 1 epoch",
                 fontsize=12.5, fontweight="bold", y=0.985, color=INK)
    olab = "MEAN(7) (dropping openbookqa)" if args.metric == "mean8" else "MEAN(8)"
    fig.text(0.5, 0.036,
             "Matched pairs: the fp16 arm is the QLoRA-merged checkpoint permuted with SALT-Q's "
             "own segmentation — columns [0:k) kept fp16, the rest GPTQ'd. It is PTQ.",
             ha="center", fontsize=8.2, color=INK2)
    fig.text(0.5, 0.010,
             f"z-only (k=0) already matches the fp16 arm at "
             f"{'5.53' if args.metric == 'mean7' else '4.26–5.53'} bits.  "
             f"On {olab} the k=256 point recovers {rec_o(sq_other[256]):.0f}%, not "
             f"{rec(sq[256][0]):.0f}% — ordering unchanged.  Bars ±1 stderr.",
             ha="center", fontsize=8.2, color=INK2)
    fig.tight_layout(rect=[0, 0.058, 1, 0.945])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(f"saved -> {args.out}   metric={mlabel}\n")

    print(f"{'group_k':>8} | {'SALT-Q 3.00 bit':>21} | {'fp16 salient':>27}")
    print(f"{'':>8} | {mlabel:>9} {'recovered':>11} | {mlabel:>9} {'recovered':>11} {'bits':>6}")
    for k in mp_ks:
        lab = "0" if k == 0 else "full" if k is None else str(k)
        s = (f"{sq[k][0]:>9.2f} {rec(sq[k][0]):>10.1f}%" if k in sq else f"{'--':>9} {'--':>11}")
        print(f"{lab:>8} | {s} | {mp[k][0]:>9.2f} {rec(mp[k][0]):>10.1f}% {MP_BITS[k]:>6.2f}")
    print(f"{'QA-LoRA':>8} | {qa:>9.2f} {rec(qa):>10.1f}% |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
