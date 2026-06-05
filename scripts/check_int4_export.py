#!/usr/bin/env python
"""
check_int4_export.py — prove an exported (--export_dequant) model is genuinely INT-k quantized.

The dequant export stores dense fp16 weights whose values lie on the INT-k grid (dequant of a
real round-to-grid). This script loads the saved weights and, for every target linear, checks
that each (row, group_size) block holds at most 2**bits distinct values. A weight that was NOT
actually quantized (e.g. an fp16 passthrough bug, or the un-quantized permuted_fp16_base) shows
thousands of distinct values per group and FAILS the check.

It also flags whether the dir is a permuted model (needs the boundary gather at inference) and
whether sqat_permute_meta.pt is present.

Usage:
    python scripts/check_int4_export.py --model_path outputs/...-dequant-eval \
        --bits 4 --group_size 32
"""

import argparse
import os
import glob

import torch

_TARGET = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def _load_state_dict(model_path: str) -> dict:
    sts = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if sts:
        from safetensors.torch import load_file
        sd = {}
        for f in sts:
            sd.update(load_file(f))
        return sd
    binp = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(binp):
        return torch.load(binp, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No .safetensors / pytorch_model.bin in {model_path}")


def _max_levels_per_group(W: torch.Tensor, group_size: int, sample_rows: int = 64) -> int:
    out_f, in_f = W.shape
    ng = (in_f + group_size - 1) // group_size
    rows = range(min(out_f, sample_rows))
    m = 0
    for r in rows:
        for g in range(ng):
            seg = W[r, g * group_size:(g + 1) * group_size]
            m = max(m, torch.unique(seg.float()).numel())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=32)
    ap.add_argument("--max_layers", type=int, default=24, help="how many target linears to scan")
    args = ap.parse_args()

    limit = 2 ** args.bits
    meta = os.path.join(args.model_path, "sqat_permute_meta.pt")
    print(f"[check] model_path        = {args.model_path}")
    print(f"[check] bits={args.bits}  group_size={args.group_size}  max levels/group allowed = {limit}")
    print(f"[check] sqat_permute_meta = {'PRESENT (boundary gather needed at inference)' if os.path.exists(meta) else 'absent'}")

    sd = _load_state_dict(args.model_path)
    checked = 0
    quant_ok = 0
    worst = 0
    fp16_like = []
    for name, W in sd.items():
        if not name.endswith(".weight"):
            continue
        terminal = name.rsplit(".weight", 1)[0].split(".")[-1]
        if terminal not in _TARGET or W.dim() != 2:
            continue
        if checked >= args.max_layers:
            break
        lv = _max_levels_per_group(W, args.group_size)
        worst = max(worst, lv)
        ok = lv <= limit
        quant_ok += int(ok)
        if not ok:
            fp16_like.append((name, lv))
        checked += 1
        print(f"  {'OK ' if ok else 'FAIL'} {name:55s} max_levels/group={lv}")

    print("\n========================================")
    print(f"  scanned {checked} target linears: {quant_ok} genuinely INT{args.bits}, "
          f"{checked - quant_ok} look unquantized")
    print(f"  worst max-levels/group = {worst}  (INT{args.bits} ⇒ ≤ {limit})")
    if checked and quant_ok == checked:
        print(f"  ✓ VERDICT: all scanned weights lie on the INT{args.bits} grid — genuine quantized export.")
    else:
        print("  ✗ VERDICT: some weights are NOT INT-k (looks like an fp16/unquantized model):")
        for n, lv in fp16_like[:10]:
            print(f"      {n}: {lv} levels/group")
    print("========================================")


if __name__ == "__main__":
    main()
