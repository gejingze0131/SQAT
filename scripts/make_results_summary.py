#!/usr/bin/env python
"""Regenerate the tables in RESULTS_SUMMARY.md from results/commonsense_vllm/*.json and
results_saltq.csv (GSM8K). Columns: method, bits, group size, the eight commonsense test sets
in the LLM-Adapters order, Avg = MEAN(8), and the share of the cell's fp16-to-GPTQ gap
recovered. Run from the repo root; prints markdown to stdout.

  python scripts/make_results_summary.py > /tmp/tables.md
"""
import csv, json, os

R = "results/commonsense_vllm"
TASKS = [("boolq", "BoolQ"), ("piqa", "PIQA"), ("siqa", "SIQA"), ("hellaswag", "HellaS."),
         ("winogrande", "WinoG."), ("arc_easy", "ARC-e"), ("arc_challenge", "ARC-c"), ("openbookqa", "OBQA")]


def load(tag):
    with open(os.path.join(R, tag + ".json")) as f:
        return json.load(f)["results"]


def row(name, bits, gs, tag, floor, ceil):
    r = load(tag)
    accs = [r[t]["acc"] * 100 for t, _ in TASKS]
    avg = sum(accs) / 8
    rec = "—" if floor is None else f"{(avg - floor) / (ceil - floor) * 100:.0f}%"
    return f"| {name} | {bits} | {gs} | " + " | ".join(f"{a:.1f}" for a in accs) + f" | **{avg:.2f}** | {rec} |"


def table(title, rows, floor_tag, ceil_tag):
    fl = sum(load(floor_tag)[t]["acc"] for t, _ in TASKS) / 8 * 100
    ce = sum(load(ceil_tag)[t]["acc"] for t, _ in TASKS) / 8 * 100
    out = [f"### {title}", "",
           "| Method | Bits | Group | " + " | ".join(n for _, n in TASKS) + " | Avg (MEAN8) | Gap recovered |",
           "|---|---|---|" + "---|" * 8 + "---|---|"]
    for name, bits, gs, tag in rows:
        out.append(row(name, bits, gs, tag, fl, ce))
    return "\n".join(out) + "\n"


SPAN3 = [("QLoRA merged fp16 (upper bound)", "16", "—", "qlora_none_cs170k_int3_ep1_span-3bit-none-merged-eval"),
         ("**SALT-Q** (k=128)", "3", "64", "saltq_cs170k_int3_g64_ep1_sal5e5_span-3bit-saltq-deploy-eval"),
         ("QA-LoRA", "3", "64", "qalora_cs170k_int3_ep1_span-3bit-qalora-dequant-eval"),
         ("fp16-salient PTQ, k=2048 (8.05 eff. bits)", "3 + fp16", "64", "mixedprec_int3_g64_span_rank_k2048"),
         ("QLoRA merged → GPTQ (floor)", "3", "64", "qlora_none_cs170k_int3_ep1_span-3bit-none-gptq-eval")]
SPAN2 = [("QLoRA merged fp16 (upper bound)", "16", "—", "qlora_none_cs170k_int2_ep1_span-2bit-none-merged-eval"),
         ("**SALT-Q** (k=256)", "2", "32", "saltq_cs170k_int2_g32_ep1_span_k256-2bit-saltq-deploy-eval"),
         ("SALT-Q (k=128)", "2", "32", "saltq_cs170k_int2_g32_ep1_span-2bit-saltq-deploy-eval"),
         ("SALT-Q (k=128, 3 epochs)", "2", "32", "saltq_cs170k_int2_g32_ep3_span-2bit-saltq-deploy-eval"),
         ("QA-LoRA", "2", "32", "qalora_cs170k_int2_ep1_span-2bit-qalora-dequant-eval"),
         ("fp16-salient PTQ, k=2048 (7.44 eff. bits)", "2 + fp16", "32", "mixedprec_int2_g32_span_rank_k2048"),
         ("QLoRA merged → GPTQ (floor)", "2", "32", "qlora_none_cs170k_int2_ep1_span-2bit-none-gptq-eval")]
RESP3 = [("QLoRA merged fp16 (upper bound)", "16", "—", "qlora_none_cs170k_int3_ep1-3bit-none-merged-eval"),
         ("**SALT-Q** (k=128)", "3", "64", "saltq_cs170k_int3_g64_ep1_sal5e5-3bit-saltq-deploy-eval"),
         ("QA-LoRA", "3", "64", "qalora_cs170k_int3_ep1-3bit-qalora-dequant-eval"),
         ("fp16-salient PTQ, k=2048 (8.05 eff. bits, index-ordered perm)", "3 + fp16", "64", "mixedprec_int3_g64_k2048"),
         ("QLoRA merged → GPTQ (floor)", "3", "64", "qlora_none_cs170k_int3_ep1-3bit-none-gptq-eval")]
RESP2 = [("QLoRA merged fp16 (upper bound)", "16", "—", "qlora_none_cs170k_int3_ep1-3bit-none-merged-eval"),
         ("SALT-Q (k=128)", "2", "32", "saltq_cs170k_int2_g32_ep1-2bit-saltq-deploy-eval"),
         ("**SALT-Q** (k=128, 3 epochs)", "2", "32", "saltq_cs170k_int2_g32_ep3-2bit-saltq-deploy-eval"),
         ("QA-LoRA", "2", "32", "qalora_cs170k_int2_ep1-2bit-qalora-dequant-eval"),
         ("QLoRA merged → GPTQ (floor)", "2", "32", "qlora_none_cs170k_int2_ep1-2bit-none-gptq-eval")]


def math_table():
    rows = [r for r in csv.DictReader(open("results_saltq.csv")) if r["task"] == "gsm8k" and "flexible" in r["metric"]]
    val = {}
    for r in rows:
        val[r["model_dir"].split("/")[-1]] = float(r["value"]) * 100
    spec = [("**SALT-Q**, AWQ fold + salient reorder (legacy [2,30], k=128)", "2", "32", "saltq_awq_legacy-2bit-saltq-deploy-eval"),
            ("SALT-Q (autoseg k=128, zp_lr 5e-3)", "2", "32", "saltq_zp5e3-2bit-saltq-deploy-eval"),
            ("QA-LoRA (lr 2e-3, tuned)", "2", "32", "qalora-int2-g32-lr2e3-2bit-qalora-dequant-eval"),
            ("QA-LoRA (lr 1e-4, paper)", "2", "32", "qalora-int2-g32-2bit-qalora-dequant-eval"),
            ("SALT-Q z-only ablation", "2", "32", "saltq_zonly-2bit-saltq-deploy-eval"),
            ("SALT-Q (k=128)", "3", "64", "saltq-3bit-saltq-deploy-eval")]
    out = ["### MetaMath → GSM8K (1 epoch, flexible-extract exact match)", "",
           "| Method | Bits | Group | GSM8K | Gap recovered |", "|---|---|---|---|---|"]
    for name, bits, gs, k in spec:
        out.append(f"| {name} | {bits} | {gs} | **{val[k]:.2f}** | — (no fp16 / GPTQ bounds run on math) |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print(table("INT3 g64, span cell", SPAN3, SPAN3[-1][3], SPAN3[0][3]))
    print(table("INT2 g32, span cell", SPAN2, SPAN2[-1][3], SPAN2[0][3]))
    print(table("INT3 g64, response-only cell (reference)", RESP3, RESP3[-1][3], RESP3[0][3]))
    print(table("INT2 g32, response-only cell (reference)", RESP2, RESP2[-1][3], RESP2[0][3]))
    print(math_table())
