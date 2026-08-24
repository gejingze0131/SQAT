#!/usr/bin/env python
"""
fp16-salient (mixed precision) vs SALT-Q, both swept over group_k, Commonsense INT3 g64.

The question: does training the salient slice replace keeping it in fp16, and how much of the
quantization gap does top-k protection actually contribute?

Both arms now have a group_k sweep, so every point on the left panel is a MATCHED pair -- same
segmentation, same salient channels, same GPTQ code path, differing only in what made the salient
slice good. SALT-Q stays at 3.00 effective bits at every k; the fp16 arm pays bits to move right.

Left  panel: MEAN(7) vs group_k, one y-axis, floor/ceiling drawn as the band being closed.
Right panel: the same runs as "% of the 4.04-point quantization gap recovered" vs what they cost
in effective bits -- where SALT-Q's four configurations stack VERTICALLY at 3.00 bits while the
fp16 arm has to walk right.

Colors are slots 1-3 of the data-viz reference palette (blue/orange/aqua), used unmodified; that
three-slot set is the documented all-pairs-validated subset in both modes.

  python scripts/plot_mixedprec_curve.py --out figures/mixedprec_curve.png
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "results/commonsense_vllm"
TASKS7 = ["arc_challenge", "arc_easy", "boolq", "piqa", "siqa", "hellaswag", "winogrande"]

# group_k -> effective bit width of the DEPLOYED fp16-salient model at INT3. o_proj is always
# group_k=0, so the fp16 share counts q/k/v/gate/up/down only. See export_mixed_precision_sweep.py.
MP_BITS = {0: 3.00, 64: 3.16, 128: 3.32, 256: 3.63, 512: 4.26, 1024: 5.53, 2048: 8.05, None: 16.00}
MP_SHARE = {0: 0.0, 64: 1.21, 128: 2.43, 256: 4.86, 512: 9.72, 1024: 19.43, 2048: 38.86,
            None: 100.0}

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


def mean7(r):
    return sum(r[t]["acc"] for t in TASKS7) / len(TASKS7) * 100


def stderr7(r):
    v = [math.sqrt(r[t]["acc"] * (1 - r[t]["acc"]) / r[t]["n"]) * 100 for t in TASKS7]
    return math.sqrt(sum(x * x for x in v)) / len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/mixedprec_curve.png")
    args = ap.parse_args()

    mp_ks = [0, 64, 128, 256, 512, 1024, 2048, None]
    mp = {}
    for k in mp_ks:
        tag = FLOOR if k == 0 else CEIL if k is None else f"mixedprec_int3_g64_k{k}"
        r = load(tag)
        mp[k] = (mean7(r), stderr7(r))
    sq = {k: (mean7(load(t)), stderr7(load(t))) for k, t in SALTQ.items()}
    qa = mean7(load(QALORA))

    floor, ceil_ = mp[0][0], mp[None][0]
    gap = ceil_ - floor
    rec = lambda a: (a - floor) / gap * 100

    plt.rcParams.update({
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#d9d8d2", "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
    })
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.9))

    # ================= left: MEAN(7) vs group_k, matched pairs =================
    xs = {k: i for i, k in enumerate(mp_ks)}

    axL.axhspan(floor, ceil_, color=INK3, alpha=0.07, zorder=0)
    for yv, lab in ((ceil_, f"fp16 merge  {ceil_:.2f}   (gap = {gap:.2f} pts)"),
                    (floor, f"GPTQ INT3 floor  {floor:.2f}")):
        axL.axhline(yv, color=INK3, lw=1.0, ls=(0, (2, 2)), zorder=1)
        axL.annotate(lab, xy=(0.04, yv + 0.09), color=INK2, fontsize=8.8)

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
    axL.annotate("3.00 bit\nthroughout", xy=(xs[256], sq[256][0]), xytext=(14, -4),
                 textcoords="offset points", fontsize=9, color=C_SQ, fontweight="bold")
    axL.annotate(f"QA-LoRA {qa:.2f}", xy=(xs[0], qa), xytext=(34, 26),
                 textcoords="offset points", fontsize=8.8, color=C_QA,
                 arrowprops=dict(arrowstyle="-", color=C_QA, lw=0.9, shrinkA=1, shrinkB=3))
    axL.annotate(f"z-only — no protected\ncolumns at all — {sq[0][0]:.2f}",
                 xy=(xs[0], sq[0][0]), xytext=(11, -30), textcoords="offset points",
                 fontsize=8.8, color=C_SQ)

    axL.set_xticks(list(xs.values()))
    axL.set_xticklabels(["0" if k == 0 else "full" if k is None else str(k) for k in mp_ks])
    axL.set_xlabel("group_k   —   leading columns given the salient treatment")
    axL.set_ylabel("Commonsense-170k  MEAN(7)   %")
    axL.set_ylim(floor - 1.05, ceil_ + 0.55)
    axL.set_title("At every matched group_k, training the salient slice beats\n"
                  "keeping it in fp16 — and costs no extra bits",
                  fontsize=11.5, fontweight="bold", color=INK)
    h, l = axL.get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: ("SALT-Q" not in l[i], "QA-LoRA" in l[i]))
    axL.legend([h[i] for i in order], [l[i] for i in order],
               loc="lower right", fontsize=9, framealpha=0.96, edgecolor="#d9d8d2")

    # ================= right: recovery vs what it costs =================
    axR.axhline(100, color=INK3, lw=1.0, ls=(0, (2, 2)), zorder=1)
    axR.annotate("fp16 merge = 100% of the gap", xy=(3.15, 101.4), color=INK2, fontsize=8.8)

    axR.plot([MP_BITS[k] for k in mp_ks], [rec(mp[k][0]) for k in mp_ks],
             color=C_MP, marker="o", ms=8, lw=2.0, zorder=3,
             label="fp16 salient + GPTQ rest")
    for k in mp_ks:
        if k in (0, None):
            continue
        axR.annotate(f"k={k}", xy=(MP_BITS[k], rec(mp[k][0])), xytext=(5, -12),
                     textcoords="offset points", fontsize=8.4, color=C_MP)

    sq_ks = sorted(sq)
    axR.plot([3.00] * len(sq_ks), [rec(sq[k][0]) for k in sq_ks],
             color=C_SQ, marker="o", ms=8, lw=2.2, zorder=4,
             label="SALT-Q  (k = 0, 64, 128, 256)")
    axR.annotate(f"k={sq_ks[0]}  (z-only)", xy=(3.00, rec(sq[sq_ks[0]][0])),
                 xytext=(9, -11), textcoords="offset points", fontsize=8.6, color=C_SQ)
    axR.annotate(f"k={sq_ks[-1]}", xy=(3.00, rec(sq[sq_ks[-1]][0])), xytext=(7, 8),
                 textcoords="offset points", fontsize=8.6, color=C_SQ)
    axR.plot([3.00], [rec(qa)], marker="s", ms=9, color=C_QA, ls="none", zorder=4,
             label="QA-LoRA")
    axR.annotate("QA-LoRA", xy=(3.00, rec(qa)), xytext=(10, -10),
                 textcoords="offset points", fontsize=8.6, color=C_QA)

    # ONE annotation: the strongest claim the panel supports. The z-only equivalence is in the
    # caption instead -- two arrows in this space collided with each other and with the curve.
    axR.annotate("", xy=(3.03, rec(sq[256][0])), xytext=(8.05, rec(sq[256][0])),
                 arrowprops=dict(arrowstyle="<->", color=C_SQ, lw=1.5,
                                 shrinkA=0, shrinkB=0))
    axR.annotate("SALT-Q k=256 at 3.00 bits is only TIED\nonce the fp16 arm reaches 8.05 bits",
                 xy=(4.9, 90.5), ha="center", fontsize=9.6, color=C_SQ, fontweight="bold")

    axR.set_xlabel("effective bit width of the deployed model")
    axR.set_ylabel("% of the 4.04-point quantization gap recovered")
    axR.set_xscale("log", base=2)
    axR.set_xlim(2.82, 18.5)
    axR.set_ylim(-7, 112)
    axR.set_xticks([3, 3.5, 4, 5, 6, 8, 11, 16])
    axR.set_xticklabels(["3", "3.5", "4", "5", "6", "8", "11", "16"])
    axR.minorticks_off()
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
    fig.text(0.5, 0.036,
             "Matched pairs: the fp16 arm is the QLoRA-merged checkpoint (82.42), permuted with "
             "SALT-Q's own segmentation, columns [0:k) kept fp16 and the rest GPTQ'd — so only the "
             "treatment of the salient slice differs.",
             ha="center", fontsize=8.3, color=INK2)
    fig.text(0.5, 0.010,
             "It is PTQ: no adaptation to the quantizer.  SALT-Q's z-only point (k=0, 3.00 bits) "
             "already matches the fp16 arm at 5.53 bits.  Error bars ±1 stderr of MEAN(7); "
             "pairwise resolution ±0.48.",
             ha="center", fontsize=8.3, color=INK2)
    fig.tight_layout(rect=[0, 0.058, 1, 0.945])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(f"saved -> {args.out}\n")

    print(f"{'group_k':>8} | {'SALT-Q 3.00bit':>21} | {'fp16 salient':>26}")
    print(f"{'':>8} | {'MEAN(7)':>9} {'recovered':>11} | {'MEAN(7)':>9} {'recovered':>11} {'bits':>5}")
    for k in mp_ks:
        lab = "0" if k == 0 else "full" if k is None else str(k)
        s = (f"{sq[k][0]:>9.2f} {rec(sq[k][0]):>10.1f}%" if k in sq else f"{'--':>9} {'--':>11}")
        print(f"{lab:>8} | {s} | {mp[k][0]:>9.2f} {rec(mp[k][0]):>10.1f}% {MP_BITS[k]:>5.2f}")
    print(f"{'QA-LoRA':>8} | {qa:>9.2f} {rec(qa):>10.1f}% |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
