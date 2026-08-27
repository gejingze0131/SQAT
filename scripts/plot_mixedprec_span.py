#!/usr/bin/env python
"""
fp16-salient (mixed precision, PTQ) vs SALT-Q vs QA-LoRA, swept over group_k, for ONE cell.

Generalises scripts/plot_mixedprec_curve.py (hard-wired to the response-only INT3 g64 cell) to
any (bit width, group size, supervision span) cell via --cell. Every input is a results JSON
written by runs/eval_vllm.sh; effective bit widths of the fp16-salient arm are read from the
mixedprec_meta.pt the sweep saves next to each exported model (fp16 share of the target
weights -> bits + share * (16 - bits)), so nothing is hand-typed.

Left panel: the metric against group_k -- matched points, since the fp16 arm and SALT-Q share
the segmentation, the salient channels and the GPTQ code path. Right panel: the same points
against the effective bit width of the DEPLOYED model, which is the honest axis: keeping k
columns in fp16 costs bits, SALT-Q and QA-LoRA sit at the nominal width whatever k is.

  python scripts/plot_mixedprec_span.py --cell int2_g32_span --metric mean7
  python scripts/plot_mixedprec_span.py --cell int3_g64_span --metric mean7
  python scripts/plot_mixedprec_span.py --cell int3_g64_resp --metric mean8   # reproduces the old figure's data
"""
import argparse, json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

RESULTS = "results/commonsense_vllm"
TASKS8 = ["arc_challenge", "arc_easy", "openbookqa", "boolq", "piqa", "siqa", "hellaswag", "winogrande"]
TASKS7 = [t for t in TASKS8 if t != "openbookqa"]

CELLS = {
    "int3_g64_resp": dict(
        bits=3, gs=64, title="Commonsense INT3 g64, response-only supervision",
        mp_prefix="mixedprec_int3_g64", mp_root="outputs/mixedprec_int3_g64",
        ks=[64, 128, 256, 512, 1024, 2048],
        floor="qlora_none_cs170k_int3_ep1-3bit-none-gptq-eval",
        ceil="qlora_none_cs170k_int3_ep1-3bit-none-merged-eval",
        qalora="qalora_cs170k_int3_ep1-3bit-qalora-dequant-eval",
        saltq={0: "saltq_cs170k_int3_g64_ep1_zonly-3bit-saltq-deploy-eval",
               64: "saltq_cs170k_int3_ep1_k64-3bit-saltq-deploy-eval",
               128: "saltq_cs170k_int3_g64_ep1_sal5e5-3bit-saltq-deploy-eval",
               256: "saltq_cs170k_int3_ep1_k256-3bit-saltq-deploy-eval"}),
    "int2_g32_span": dict(
        bits=2, gs=32, title="Commonsense INT2 g32, instruction+response supervision",
        mp_prefix="mixedprec_int2_g32_span", mp_root="outputs/mixedprec_int2_g32_span",
        ks=[32, 64, 128, 256, 512, 1024, 2048],
        floor="qlora_none_cs170k_int2_ep1_span-2bit-none-gptq-eval",
        ceil="qlora_none_cs170k_int2_ep1_span-2bit-none-merged-eval",
        qalora="qalora_cs170k_int2_ep1_span-2bit-qalora-dequant-eval",
        saltq={128: "saltq_cs170k_int2_g32_ep1_span-2bit-saltq-deploy-eval"}),
    "int2_g32_span_rank": dict(
        bits=2, gs=32, title="Commonsense INT2 g32, instruction+response supervision (rank-ordered top-k)",
        mp_prefix="mixedprec_int2_g32_span_rank", mp_root="outputs/mixedprec_int2_g32_span_rank",
        ks=[32, 64, 128, 256, 512, 1024, 2048],
        floor="qlora_none_cs170k_int2_ep1_span-2bit-none-gptq-eval",
        ceil="qlora_none_cs170k_int2_ep1_span-2bit-none-merged-eval",
        qalora="qalora_cs170k_int2_ep1_span-2bit-qalora-dequant-eval",
        saltq={128: "saltq_cs170k_int2_g32_ep1_span-2bit-saltq-deploy-eval"}),
    "int3_g64_span_rank": dict(
        bits=3, gs=64, title="Commonsense INT3 g64, instruction+response supervision (rank-ordered top-k)",
        mp_prefix="mixedprec_int3_g64_span_rank", mp_root="outputs/mixedprec_int3_g64_span_rank",
        ks=[64, 128, 256, 512, 1024, 2048],
        floor="qlora_none_cs170k_int3_ep1_span-3bit-none-gptq-eval",
        ceil="qlora_none_cs170k_int3_ep1_span-3bit-none-merged-eval",
        qalora="qalora_cs170k_int3_ep1_span-3bit-qalora-dequant-eval",
        saltq={128: "saltq_cs170k_int3_g64_ep1_sal5e5_span-3bit-saltq-deploy-eval"}),
    "int3_g64_span": dict(
        bits=3, gs=64, title="Commonsense INT3 g64, instruction+response supervision",
        mp_prefix="mixedprec_int3_g64_span", mp_root="outputs/mixedprec_int3_g64_span",
        ks=[64, 128, 256, 512, 1024, 2048],
        floor="qlora_none_cs170k_int3_ep1_span-3bit-none-gptq-eval",
        ceil="qlora_none_cs170k_int3_ep1_span-3bit-none-merged-eval",
        qalora="qalora_cs170k_int3_ep1_span-3bit-qalora-dequant-eval",
        saltq={128: "saltq_cs170k_int3_g64_ep1_sal5e5_span-3bit-saltq-deploy-eval"}),
}

C_SQ, C_MP, C_QA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"


def load(tag):
    with open(os.path.join(RESULTS, f"{tag}.json")) as f:
        return json.load(f)["results"]


def mp_meta(cell, k):
    """(fp16 share of target weights, effective bit width) from the sweep's own meta; None if
    that point was not exported."""
    p = os.path.join(cell["mp_root"], f"k{k}", "mixedprec_meta.pt")
    if not os.path.exists(p):
        return None
    m = torch.load(p, map_location="cpu", weights_only=False)
    share = float(m["fp16_share"])
    return share, float(m.get("effective_bits") or (cell["bits"] + share * (16 - cell["bits"])))


def eff_bits(cell, k):
    m = mp_meta(cell, k)
    return None if m is None else m[1]


def plot_share(cell, args, mp, sq, qa, floor, ceil_, gap, rec, mlabel, out):
    """Single panel, x = the TRUE fraction of target weights kept in fp16 (linear). The GPTQ floor
    is share 0, the fp16 merge is share 100%; SALT-Q and QA-LoRA keep nothing in fp16 and sit at
    x = 0 as distinct markers. Effective bit widths are written beside each fp16-salient point."""
    bits = cell["bits"]
    pts = []
    for k in cell["ks"]:
        if k in mp and mp_meta(cell, k):
            sh, eb = mp_meta(cell, k)
            pts.append((sh * 100, mp[k][0], mp[k][1], k, eb))
    pts.sort()
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.axhspan(floor, ceil_, color=INK3, alpha=0.07, zorder=0)
    for yv, lab in ((ceil_, f"fp16 merge (100% fp16)  {ceil_:.2f}"), (floor, f"GPTQ INT{bits} g{cell['gs']} floor (0% fp16)  {floor:.2f}")):
        ax.axhline(yv, color=INK3, lw=1.0, ls=(0, (2, 2)), zorder=1)
        ax.annotate(lab, xy=(0.5, yv + 0.012 * gap), xycoords=("axes fraction", "data"), color=INK2, fontsize=8.8)
    xs = [0.0] + [p[0] for p in pts]; ys = [floor] + [p[1] for p in pts]; es = [mp[0][1]] + [p[2] for p in pts]
    ax.errorbar(xs, ys, yerr=es, color=C_MP, marker="o", ms=7, lw=2.0, capsize=3, zorder=3,
                label="fp16 salient + GPTQ rest  (PTQ, top-k by saliency)")
    for sh, y, _, k, eb in pts:
        ax.annotate(f"k={k}\n{eb:.2f} b", xy=(sh, y), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=7.6, color=C_MP)
    stack = sorted([(sq[k][0], f"SALT-Q k={k}  {sq[k][0]:.2f}  ({rec(sq[k][0]):.0f}% of gap)", C_SQ, sq[k][1], "o") for k in sq]
                   + [(qa[0], f"QA-LoRA  {qa[0]:.2f}  ({rec(qa[0]):.0f}%)", C_QA, qa[1], "s")], key=lambda t: t[0])
    mid = (len(stack) - 1) / 2
    for i, (y, lab, col, e, mk) in enumerate(stack):
        ax.errorbar([0.0], [y], yerr=[e], color=col, marker=mk, ms=9, capsize=3, zorder=4, ls="none",
                    label=("SALT-Q" if mk == "o" else "QA-LoRA") + f"  (0% fp16, {bits}.00 bits)" if i == 0 or mk == "s" else None)
        ax.annotate(lab, xy=(0.0, y), xytext=(14, (i - mid) * 11), textcoords="offset points", fontsize=8.5, color=col, va="center")
    ax.set_xlim(-1.5, max(41, max(xs) * 1.06))
    ax.set_ylim(floor - 0.06 * gap, ceil_ + 0.12 * gap)
    ax.set_xlabel("share of target weights kept in fp16  (%, true proportion)")
    ax.set_ylabel(mlabel)
    ax.set_title(f"{cell['title']}\n{mlabel} vs the true fp16 share", fontsize=10.5)
    ax.legend(loc="best", fontsize=8.6, framealpha=0.95)
    fig.text(0.5, 0.005, "fp16 arm = the cell's QLoRA-merged checkpoint, top-k columns (by the saved saliency ranking) kept in fp16, rest GPTQ. "
             "Error bars: binomial stderr propagated to the mean. Recovery = (score - floor) / (fp16 - floor).", ha="center", fontsize=7.6, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=sorted(CELLS), required=True)
    ap.add_argument("--metric", choices=["mean8", "mean7"], default="mean8")
    ap.add_argument("--out", default=None)
    ap.add_argument("--x", choices=["groupk", "share"], default="groupk",
                    help="share = single panel with the true fp16 proportion on x")
    args = ap.parse_args()
    cell = CELLS[args.cell]
    bits = cell["bits"]
    tasks = TASKS8 if args.metric == "mean8" else TASKS7
    mlabel = f"MEAN({len(tasks)})"
    out = args.out or f"figures/mixedprec_{args.cell}_{args.metric}{'_share' if args.x == 'share' else ''}.png"

    def agg(r): return sum(r[t]["acc"] for t in tasks) / len(tasks) * 100
    def err(r):
        v = [math.sqrt(r[t]["acc"] * (1 - r[t]["acc"]) / r[t]["n"]) * 100 for t in tasks]
        return math.sqrt(sum(x * x for x in v)) / len(v)

    # fp16-salient arm: k=0 is the GPTQ floor, k=None the fp16 merge. Missing sweep points are
    # dropped (and reported) rather than faked.
    mp, mpb, missing = {}, {}, []
    for k in [0] + cell["ks"] + [None]:
        tag = cell["floor"] if k == 0 else cell["ceil"] if k is None else f"{cell['mp_prefix']}_k{k}"
        try:
            r = load(tag)
        except FileNotFoundError:
            missing.append(k); continue
        mp[k] = (agg(r), err(r))
        mpb[k] = float(bits) if k == 0 else 16.0 if k is None else eff_bits(cell, k)
    if missing:
        print(f"[plot] {args.cell}: missing sweep points {missing} -- plotted without them")
    sq = {k: (agg(load(t)), err(load(t))) for k, t in cell["saltq"].items() if os.path.exists(os.path.join(RESULTS, t + ".json"))}
    qa_r = load(cell["qalora"]); qa = (agg(qa_r), err(qa_r))
    floor, ceil_ = mp[0][0], mp[None][0]
    gap = ceil_ - floor
    rec = lambda a: (a - floor) / gap * 100

    plt.rcParams.update({
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#d9d8d2", "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
    })
    if args.x == "share":
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plot_share(cell, args, mp, sq, qa, floor, ceil_, gap, rec, mlabel, out)
        return

    plt.rcParams.update({
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#d9d8d2", "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
    })
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.9))
    ks_all = [k for k in [0] + cell["ks"] + [None] if k in mp]
    xs = {k: i for i, k in enumerate(ks_all)}
    xlab = ["GPTQ\n(k=0)"] + [str(k) for k in ks_all[1:-1]] + ["fp16\nmerge"]

    # ---------------- left: metric vs group_k
    axL.axhspan(floor, ceil_, color=INK3, alpha=0.07, zorder=0)
    for yv, lab in ((ceil_, f"fp16 merge  {ceil_:.2f}   (gap = {gap:.2f} pts)"),
                    (floor, f"GPTQ INT{bits} g{cell['gs']} floor  {floor:.2f}")):
        axL.axhline(yv, color=INK3, lw=1.0, ls=(0, (2, 2)), zorder=1)
        axL.annotate(lab, xy=(0.04, yv + 0.012 * gap), color=INK2, fontsize=8.8)
    axL.errorbar([xs[k] for k in ks_all], [mp[k][0] for k in ks_all], yerr=[mp[k][1] for k in ks_all],
                 color=C_MP, marker="o", ms=8, lw=2.0, capsize=3, zorder=3,
                 label="fp16 salient + GPTQ rest  (PTQ, pays bits)")
    sq_ks = [k for k in sorted(sq) if k in xs]
    if sq_ks:
        axL.errorbar([xs[k] for k in sq_ks], [sq[k][0] for k in sq_ks], yerr=[sq[k][1] for k in sq_ks],
                     color=C_SQ, marker="o", ms=9, lw=2.2, capsize=3, zorder=4, ls="-" if len(sq_ks) > 1 else "none",
                     label=f"SALT-Q trained salient  (always {bits:.2f} bits)")
    axL.errorbar([xs[0]], [qa[0]], yerr=[qa[1]], marker="s", ms=9, color=C_QA, ls="none", capsize=3,
                 zorder=4, label=f"QA-LoRA  ({bits:.2f} bits)")
    for k in ks_all[1:-1]:
        if mpb.get(k):
            axL.annotate(f"{mpb[k]:.2f} b", xy=(xs[k], mp[k][0]), xytext=(0, -16), textcoords="offset points",
                         ha="center", fontsize=8, color=C_MP)
    axL.set_xticks(range(len(ks_all))); axL.set_xticklabels(xlab)
    axL.set_xlabel("group_k  (salient columns per layer kept in fp16 / trained)")
    axL.set_ylabel(mlabel)
    axL.set_ylim(floor - 0.06 * gap, ceil_ + 0.10 * gap)
    axL.set_title(f"{cell['title']}\n{mlabel} vs group_k (matched segmentation, matched GPTQ)", fontsize=10.5)
    axL.legend(loc="best", fontsize=8.8, framealpha=0.95)

    # ---------------- right: metric vs effective bits
    pts = [(mpb[k], mp[k][0], mp[k][1]) for k in ks_all if mpb.get(k)]
    pts.sort()
    axR.axhspan(floor, ceil_, color=INK3, alpha=0.07, zorder=0)
    axR.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], color=C_MP,
                 marker="o", ms=7, lw=1.8, capsize=3, zorder=3, label="fp16 salient + GPTQ rest")
    # Everything at the nominal width shares one x; stack the labels by value so they never
    # overlap (each label gets its own row, 11 pt apart, ordered like the points).
    stack = sorted([(sq[k][0], f"SALT-Q k={k}  {sq[k][0]:.2f}  ({rec(sq[k][0]):.0f}% of gap)", C_SQ) for k in sq_ks]
                   + [(qa[0], f"QA-LoRA  {qa[0]:.2f}  ({rec(qa[0]):.0f}%)", C_QA)], key=lambda t: t[0])
    for k in sq_ks:
        axR.errorbar([bits], [sq[k][0]], yerr=[sq[k][1]], color=C_SQ, marker="o", ms=9, capsize=3, zorder=4,
                     label="SALT-Q" if k == sq_ks[0] else None)
    axR.errorbar([bits], [qa[0]], yerr=[qa[1]], color=C_QA, marker="s", ms=9, capsize=3, zorder=4, label="QA-LoRA")
    mid = (len(stack) - 1) / 2
    for i, (y, lab, col) in enumerate(stack):
        axR.annotate(lab, xy=(bits, y), xytext=(14, (i - mid) * 11), textcoords="offset points",
                     fontsize=8.5, color=col, va="center")
    for b, y, _ in pts:
        if bits < b < 16:
            axR.annotate(f"{rec(y):.0f}%", xy=(b, y), xytext=(0, 8), textcoords="offset points", ha="center",
                         fontsize=7.5, color=C_MP)
    axR.set_xscale("log", base=2)
    axR.set_xticks([2, 3, 4, 6, 8, 12, 16]); axR.set_xticklabels(["2", "3", "4", "6", "8", "12", "16"])
    axR.set_xlabel("effective bit width of the deployed model (log scale)")
    axR.set_ylabel(mlabel)
    axR.set_ylim(floor - 0.06 * gap, ceil_ + 0.10 * gap)
    axR.set_title("Read per bit spent: fp16-salient must buy width to climb;\nSALT-Q and QA-LoRA stay at the nominal width", fontsize=10.5)
    axR.legend(loc="lower right", fontsize=8.8, framealpha=0.95)

    fig.text(0.5, 0.005,
             f"fp16 arm = the cell's QLoRA-merged checkpoint permuted with SALT-Q's saved segmentation, top-k columns kept in fp16, "
             f"rest GPTQ INT{bits} g{cell['gs']}. Error bars: binomial stderr propagated to the mean. Recovery = (score - floor) / (fp16 - floor).",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")

    print(f"\n{args.cell}  {mlabel}   floor {floor:.2f}   fp16 {ceil_:.2f}   gap {gap:.2f}")
    print(f"{'group_k':>8} | {'fp16-salient':>12} {'eff bits':>9} {'recov':>7} | {'SALT-Q':>8} {'recov':>7}")
    for k in ks_all:
        m = mp[k]; s = sq.get(k)
        kl = "0/GPTQ" if k == 0 else "fp16" if k is None else str(k)
        print(f"{kl:>8} | {m[0]:12.2f} {mpb.get(k) or float('nan'):9.2f} {rec(m[0]):6.1f}% | "
              + (f"{s[0]:8.2f} {rec(s[0]):6.1f}%" if s else f"{'':8s} {'':7s}"))
    print(f"{'QA-LoRA':>8} | {'':12s} {bits:9.2f} {'':7s} | {qa[0]:8.2f} {rec(qa[0]):6.1f}%")


if __name__ == "__main__":
    main()
