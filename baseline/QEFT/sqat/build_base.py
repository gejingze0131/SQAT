"""
Stage 0 of the QEFT baseline: build the mixed-precision base its fine-tuning starts from.

    calibrate λ = diag(2 X X^T)  →  one global weak-column set (OGR)  →  fold the permutation
    into the whole network  →  GPTQ everything except the weak columns  →  dense checkpoint

One GPU, one process, ~1 h for Llama-2-7B at 3500 calibration records. Idempotent: it refuses to
overwrite a base whose meta matches the config, and refuses to REUSE one whose meta does not —
a trained checkpoint only means anything against the exact base it was trained on.

    python baseline/QEFT/sqat/build_base.py --config configs/qeft_cs170k_int3_g64_ep1_span_bcal.yaml
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qeft_common import (QEFT_META_FILENAME, DEFAULT_TARGETS, build_calibration_loader,  # noqa: E402
                         build_qeft_base, load_qeft_meta)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.model_loader import load_tokenizer  # noqa: E402


def base_settings(cfg: dict) -> dict:
    q = cfg["qeft"]
    return {
        "model_name": cfg["model"]["name"],
        "k": int(q["k"]),
        "group_size": int(q["group_size"]),
        "q_bits": int(cfg["model"]["quant_bits"]),
        "symmetric": bool(cfg["qat"].get("symmetric", False)),
        "targets": list(q.get("targets", DEFAULT_TARGETS)),
        "oproj_weak": bool(q.get("oproj_weak", True)),
    }


def _matches(meta: dict, want: dict) -> list:
    """Which settings of an existing base disagree with the config."""
    bad = []
    for key, val in want.items():
        have = meta.get(key)
        if key in ("targets",):
            have, val = list(have or []), list(val)
        if have != val:
            bad.append(f"{key}: base has {have!r}, config wants {val!r}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_dir", default=None, help="overrides qeft.base_dir")
    ap.add_argument("--force", action="store_true", help="rebuild even if the base exists")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    base_dir = args.base_dir or cfg["qeft"]["base_dir"]
    want = base_settings(cfg)

    if os.path.exists(os.path.join(base_dir, QEFT_META_FILENAME)) and not args.force:
        bad = _matches(load_qeft_meta(base_dir), want)
        if bad:
            print(f"ERROR: {base_dir} already holds a QEFT base built for another configuration:",
                  file=sys.stderr)
            for line in bad:
                print(f"  - {line}", file=sys.stderr)
            print("  Point qeft.base_dir somewhere else, or pass --force to overwrite it. "
                  "Overwriting invalidates every checkpoint trained against it: the weak-column "
                  "SET is a one-shot discrete choice, so a rebuilt base makes the saved columns "
                  "land on different channels.", file=sys.stderr)
            return 2
        print(f"[QEFT] base already at {base_dir} (settings match) — skipping")
        return 0

    dtype = getattr(torch, cfg["model"].get("dtype", "bfloat16"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg)
    cal_loader = build_calibration_loader(cfg, tokenizer)
    sq = cfg["qat"]["sqat"]
    gq = cfg["qat"].get("gptq", {}) or {}
    nsamples = int(gq.get("nsamples", sq["calibration_samples"]))
    if nsamples != int(sq["calibration_samples"]):
        print(f"WARNING: qat.gptq.nsamples={nsamples} != qat.sqat.calibration_samples="
              f"{sq['calibration_samples']}. GPTQ stops after the first nsamples records, so the "
              f"Hessians would see a different (and smaller) set than the λ statistics.")

    print("=" * 78)
    print(f"  QEFT base — INT{want['q_bits']} g{want['group_size']}, k={want['k']} weak columns, "
          f"{'asym' if not want['symmetric'] else 'sym'}")
    print(f"  model       {want['model_name']}")
    print(f"  calibration {sq['calibration_samples']} records, sampling="
          f"{sq.get('calibration_sampling', 'first')}, seq_len={sq['calibration_seq_len']}")
    print(f"  out         {base_dir}")
    print("=" * 78)

    build_qeft_base(
        want["model_name"], tokenizer, cal_loader, base_dir,
        k=want["k"], group_size=want["group_size"], q_bits=want["q_bits"],
        symmetric=want["symmetric"], targets=want["targets"], oproj_weak=want["oproj_weak"],
        percdamp=float(gq.get("percdamp", 0.01)),
        blocksize=int(gq.get("blocksize", 128)),
        nsamples=nsamples,
        dtype=dtype, device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
