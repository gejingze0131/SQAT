#!/usr/bin/env python
"""
packed_wq must be a STORAGE change and nothing else.

WHY THIS EXISTS. SALTQLinear precomputes `wq = q*s` as a dense bf16 buffer so the frozen forward
is one GEMM. At 7B that is 12 GiB and obviously worth it; at Qwen2.5-32B it is 58.1 GiB against a
48 GB card — 16 bits per rank holding a 2-bit value, and more than QLoRA's whole NF4 base for the
same model. packed_wq keeps the codes packed on device (7.3 GiB) and rebuilds wq per forward.

The only acceptable outcome is BIT-IDENTICAL: the rebuild repeats the precompute's exact
arithmetic (unpack -> fp32 -> multiply by saltq_s in fp32 -> one cast to wq_dtype). If it were
merely close, every SALT-Q number on this branch would be incomparable to the 7B rows in
results_saltq.csv for a reason having nothing to do with the method.

Checks:
  1. _pack_codes / _unpack_codes round-trip exactly, at every bit width the grid can take,
     asymmetric and symmetric (symmetric codes are negative — they are shifted by Qn);
  2. the rebuilt wq is bit-identical to the precomputed buffer;
  3. forward() is bit-identical, packed vs not, and so are the GRADIENTS reaching z and the
     salient weights (a storage change must not move the backward either);
  4. deployed_tensors() is bit-identical — export reads codes through _codes_ref(), which under
     packed_wq unpacks from the device buffer because there is no host copy;
  5. the host copy really is gone, and the device buffer really is ~q_bits/16 of the bf16 one.

Run:  python scripts/test_saltq_packed_wq.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.qat_saltq import SALTQLinear, _lsq_levels, _pack_codes, _unpack_codes

OUT, IN, GK, GS = 64, 256, 32, 32


def _make(packed, q_bits=2, symmetric=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    qn, qp = _lsq_levels(q_bits, symmetric)
    n_nonsal = IN - GK
    codes = torch.randint(qn, qp + 1, (OUT, n_nonsal), generator=g).to(torch.int8)
    s_n = torch.rand(OUT, n_nonsal // GS, generator=g) * 0.05 + 0.01
    z_n = torch.rand(OUT, n_nonsal // GS, generator=g) * (qp - qn) + qn
    return SALTQLinear(
        in_features=IN, out_features=OUT, group_k=GK, group_size=GS,
        q_bits=q_bits, symmetric=symmetric,
        codes=codes, s_n=s_n, z_n=z_n,
        w_s=torch.randn(OUT, GK, generator=g) * 0.02,
        s_s=torch.rand(OUT, GK // GS, generator=g) * 0.05 + 0.01,
        z_s=torch.rand(OUT, GK // GS, generator=g) * (qp - qn) + qn,
        packed_wq=packed,
    )


def main():
    fail = 0

    # 1. pack/unpack round-trip
    bad, density = [], {}
    for q_bits in (2, 3, 4):
        for symmetric in (False, True):
            qn, qp = _lsq_levels(q_bits, symmetric)
            g = torch.Generator().manual_seed(q_bits * 2 + symmetric)
            c = torch.randint(qn, qp + 1, (8, 4 * 24), generator=g).to(torch.int8)
            pk, is_pk = _pack_codes(c, q_bits, qn)
            back = _unpack_codes(pk, q_bits, qn, is_pk)
            if not torch.equal(back, c):
                bad.append(f"b{q_bits}{'sym' if symmetric else 'asym'}")
            density[q_bits] = (is_pk, c.numel() / pk.numel())
    ok = not bad
    print(f"  [{'OK  ' if ok else 'FAIL'}] pack/unpack round-trip exact at INT2/3/4, asym+sym"
          f"{'' if ok else ' — broken: ' + ','.join(bad)}")
    fail += not ok

    # 1b. every bit width the grid can take is ACTUALLY packed, not silently on the int8 fallback.
    # INT3 does not divide 8 and gets 2 codes/byte (bits 6-7 unused); it is the one that regressed
    # to 1 code/byte before, which at 32B is 29.1 GiB of resident codes instead of 14.5.
    want = {2: 4.0, 3: 2.0, 4: 2.0}
    dbad = [f"b{b}({'unpacked' if not density[b][0] else f'{density[b][1]:g}/byte'})"
            for b in want if not density[b][0] or density[b][1] != want[b]]
    ok = not dbad
    print(f"  [{'OK  ' if ok else 'FAIL'}] codes/byte is 4 / 2 / 2 at INT2 / INT3 / INT4"
          f"{'' if ok else ' — wrong: ' + ', '.join(dbad)}")
    fail += not ok

    # 2 + 3. wq, forward and gradients, bit-identical
    ref, pkd = _make(False), _make(True)
    ok = torch.equal(ref.wq, pkd._wq_tensor())
    print(f"  [{'OK  ' if ok else 'FAIL'}] rebuilt wq is bit-identical to the precomputed buffer")
    fail += not ok

    torch.manual_seed(1)
    # bf16, because wq is bf16 and F.linear requires matching dtypes — that is the layer's own
    # pre-existing contract (the model runs in bfloat16), not anything packed_wq introduces.
    x = torch.randn(3, 17, IN, dtype=torch.bfloat16)
    ya, yb = ref(x.clone()), pkd(x.clone())
    ok = torch.equal(ya, yb)
    print(f"  [{'OK  ' if ok else 'FAIL'}] forward bit-identical "
          f"(max |Δ| = {(ya - yb).abs().max().item():g})")
    fail += not ok

    ya.float().pow(2).sum().backward()
    yb.float().pow(2).sum().backward()
    gbad = []
    for nm in ("saltq_z", "weight_salient", "lsq_w_scale", "lsq_w_zp"):
        ga, gb = getattr(ref, nm).grad, getattr(pkd, nm).grad
        if (ga is None) != (gb is None):
            gbad.append(f"{nm}(one None)"); continue
        if ga is not None and not torch.equal(ga, gb):
            gbad.append(f"{nm}({(ga - gb).abs().max().item():g})")
    ok = not gbad
    print(f"  [{'OK  ' if ok else 'FAIL'}] gradients bit-identical for z / salient / LSQ grid"
          f"{'' if ok else ' — differ: ' + ', '.join(gbad)}")
    fail += not ok

    # 4. export path
    da, db = ref.deployed_tensors(), pkd.deployed_tensors()
    ok = all(torch.equal(a, b) for a, b in zip(da, db))
    print(f"  [{'OK  ' if ok else 'FAIL'}] deployed_tensors() bit-identical (codes/scale/zp)")
    fail += not ok

    # 5. storage actually changed
    ok = ref._codes_cpu is not None and pkd._codes_cpu is None and pkd.wq is None
    print(f"  [{'OK  ' if ok else 'FAIL'}] host code copy dropped and wq buffer gone under packed_wq")
    fail += not ok

    bf16_bytes = ref.wq.numel() * ref.wq.element_size()
    pk_bytes = pkd.wq_codes.numel() * pkd.wq_codes.element_size()
    ok = pk_bytes < bf16_bytes / 6                     # INT2: 4 codes/byte vs 2 bytes/code = 1/8
    print(f"  [{'OK  ' if ok else 'FAIL'}] device buffer {pk_bytes} B vs bf16 wq {bf16_bytes} B "
          f"({bf16_bytes / pk_bytes:.1f}x smaller)")
    fail += not ok

    print("\nPASS — packed_wq is storage-only." if not fail else f"\nFAIL — {fail} check(s)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
