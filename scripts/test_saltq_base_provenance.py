#!/usr/bin/env python
"""
Regression test for the two guards that stop a SALT-Q run from training on scrambled codes.

WHAT WENT WRONG. The frozen GPTQ codes are integers at FIXED COLUMN POSITIONS of the permuted
weight. Nothing in the file format records which permutation produced those positions, and the
reuse check only compared (bits, group_size, symmetric). So the INT3 g64 commonsense run
regenerated its permuted base (the calibration data had changed under the new prompt pipeline)
and reused codes from the previous permutation: every quantization group ended up holding
columns that never belonged together. Reconstruction error 114% — worse than emitting zeros —
untrained loss 10.6 against ln(32000) = 10.37, and the code histograms, the loss curve and the
export all looked completely normal.

  1. the fingerprint separates two permutations that differ in a SINGLE index
  2. ... and matches an identical one, so 1 cannot pass vacuously
  3. tensor-valued and list-valued perms fingerprint the same (perm_meta carries both forms)
  4. a stale base on disk is refused, a matching one is accepted
  5. the reconstruction backstop passes a real INT3 error (60%) and rejects the scramble (114%)

Run: python scripts/test_saltq_base_provenance.py
"""

import copy
import importlib.util
import os
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.qat_saltq import SALTQ_META_FILENAME, _assert_codes_reconstruct


def _load_train_module():
    """scripts/train.py is a script, not a package member — load it by path."""
    spec = importlib.util.spec_from_file_location("_train", os.path.join(REPO, "scripts", "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_perm_meta(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "boundary_sizes": [1, 1, 10, 20],
        "layer_group_ks": [128] * 32,
        "down_layer_group_ks": [128] * 32,
        "segment_group_ks": [128] * 4,
        "segment_perms": {k: torch.randperm(64, generator=g).tolist() for k in range(4)},
        "block_internal_perms": {f"{l}_0": torch.randperm(32, generator=g).tolist()
                                 for l in range(4)},
    }


def write_base(dir_path: str, perm_meta: dict) -> str:
    os.makedirs(dir_path, exist_ok=True)
    torch.save({"q_bits": 3, "group_size": 64, "symmetric": False, "perm_meta": perm_meta},
               os.path.join(dir_path, SALTQ_META_FILENAME))
    return dir_path


def main() -> int:
    t = _load_train_module()
    fp = t._permutation_fingerprint

    a = make_perm_meta(0)
    b = make_perm_meta(1)

    # --- 1. a single moved index changes the fingerprint --------------------------------------
    nudged = copy.deepcopy(a)
    nudged["segment_perms"][2][0], nudged["segment_perms"][2][1] = (
        nudged["segment_perms"][2][1], nudged["segment_perms"][2][0])
    assert fp(a) != fp(nudged), "swapping two columns of one segment did not change the fingerprint"
    assert fp(a) != fp(b), "two independent permutations share a fingerprint"
    print(f"[1] one swapped index                  {fp(a)} != {fp(nudged)}")

    # --- 2. an identical permutation matches ---------------------------------------------------
    assert fp(a) == fp(copy.deepcopy(a)), "a permutation does not match itself"
    print(f"[2] identical permutation              {fp(a)} == {fp(copy.deepcopy(a))}")

    # --- 3. tensors and lists agree ------------------------------------------------------------
    as_tensor = copy.deepcopy(a)
    as_tensor["segment_perms"] = {k: torch.tensor(v) for k, v in a["segment_perms"].items()}
    as_tensor["layer_group_ks"] = torch.tensor(a["layer_group_ks"])
    assert fp(a) == fp(as_tensor), (
        "the same permutation fingerprints differently as a tensor than as a list, so a base "
        "would be needlessly rebuilt (or worse, a stale one silently accepted)")
    print(f"[3] tensor form == list form           {fp(as_tensor)}")

    # --- 4. the on-disk check ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        good = write_base(os.path.join(tmp, "good"), a)
        stale = write_base(os.path.join(tmp, "stale"), b)
        assert t._saltq_base_permutation_matches(good, a), "a base built against `a` was refused"
        assert not t._saltq_base_permutation_matches(stale, a), (
            "a base built against a DIFFERENT permutation was accepted — this is exactly the "
            "combination that produced the 114% reconstruction error")
        assert not t._saltq_base_permutation_matches(os.path.join(tmp, "absent"), a)
        # A base predating this field has no provenance to check, and reusing it is the same
        # gamble that failed. It must be refused, not waved through.
        legacy = write_base(os.path.join(tmp, "legacy"), {})
        assert not t._saltq_base_permutation_matches(legacy, a), (
            "a base carrying no perm_meta was accepted on trust")
        print("[4] on-disk provenance: matching accepted, stale/missing/legacy refused")

    # --- 5. the reconstruction backstop --------------------------------------------------------
    healthy = [(f"layer{i}", 0.60) for i in range(8)]
    _assert_codes_reconstruct(healthy, "<base>", "<permuted>")   # must not raise
    scrambled = [(f"layer{i}", 1.14) for i in range(8)]
    try:
        _assert_codes_reconstruct(scrambled, "<base>", "<permuted>")
    except RuntimeError as exc:
        assert "114" in str(exc)
        print("[5] backstop: 60% accepted, 114% refused")
    else:
        raise AssertionError(
            "the backstop accepted a 114% reconstruction error — a dequantized weight further "
            "from the target than zero is, which no bit width produces")

    print("\nPASS — stale frozen codes cannot reach training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
