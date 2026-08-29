"""
Derive a salient_init="gptq_latent" SALT-Q base OFFLINE from two existing bases that share one
permutation:
  --codes_base : a salient_init="gptq" base (whole-matrix OBS codes, grids, dequantized salient start)
  --fp16_base  : any base built on the SAME permuted base whose `.w_s` is the fp16 salient slice
                 (salient_init="minmax")
The result is byte-identical to --codes_base except `.w_s`, which keeps the fp16 value wherever
RTN of it on the sweep's own grid already equals the GPTQ code and takes the GPTQ cell centre elsewhere — exactly
what build_saltq_base(salient_init="gptq_latent") produces, without re-running the 45-minute
sweep. Every code is asserted unchanged; the perm_meta of the two inputs is asserted equal.
"""
import argparse, os, shutil, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safetensors import safe_open
from safetensors.torch import save_file
from src.qat_saltq import SALTQ_BASE_FILENAME, SALTQ_META_FILENAME
from src.quant_primitives import group_quantize

def _same(a, b, path=""):
    if isinstance(a, dict):
        return set(a) == set(b) and all(_same(a[k], b[k], path + f"{k}/") for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, path) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.shape == b.shape and torch.equal(a, b)
    return a == b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes_base", required=True); ap.add_argument("--fp16_base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mc = torch.load(os.path.join(a.codes_base, SALTQ_META_FILENAME), map_location="cpu", weights_only=False)
    mf = torch.load(os.path.join(a.fp16_base, SALTQ_META_FILENAME), map_location="cpu", weights_only=False)
    assert mc.get("salient_init") == "gptq", f"--codes_base must be a salient_init=gptq base, got {mc.get('salient_init')}"
    assert mf.get("salient_init", "minmax") == "minmax", "--fp16_base must be a minmax base (fp16 salient slice)"
    for k in ("q_bits", "group_size", "symmetric", "train_salient", "target_terminals"):
        assert mc[k] == mf[k], f"meta mismatch on {k}: {mc[k]} vs {mf[k]}"
    assert _same(mc["perm_meta"], mf["perm_meta"]), "the two bases do not share a permutation"
    bits, gs, sym = int(mc["q_bits"]), int(mc["group_size"]), bool(mc["symmetric"])
    if os.path.exists(os.path.join(a.out, SALTQ_META_FILENAME)):
        raise SystemExit(f"{a.out} already holds a base; refusing to overwrite")
    tensors = {}
    with safe_open(os.path.join(a.codes_base, SALTQ_BASE_FILENAME), "pt") as fc, \
         safe_open(os.path.join(a.fp16_base, SALTQ_BASE_FILENAME), "pt") as ff:
        n_tot = n_kept = 0
        for k in fc.keys():
            t = fc.get_tensor(k)
            if k.endswith(".w_s"):
                name = k[:-4]
                s0 = fc.get_tensor(f"{name}.s_s").float(); z0 = None if sym else fc.get_tensor(f"{name}.z_s").float()
                fs = s0 if sym else (s0, z0)
                w_deq = t.float(); w_fp = ff.get_tensor(k).float()
                assert w_fp.shape == w_deq.shape, name
                q_g, _, _ = group_quantize(w_deq, gs, bits, sym, fixed_scale=fs)   # the GPTQ codes
                q_rtn, _, _ = group_quantize(w_fp, gs, bits, sym, fixed_scale=fs)    # RTN of fp16 on the sweep's grid
                keep = q_rtn == q_g
                w_new = torch.where(keep, w_fp, w_deq).contiguous()
                q_chk, _, _ = group_quantize(w_new, gs, bits, sym, fixed_scale=fs)
                assert torch.equal(q_chk, q_g), f"{name}: codes changed by the latent start"
                n_kept += int(keep.sum()); n_tot += keep.numel()
                t = w_new
            tensors[k] = t.contiguous()
    os.makedirs(a.out, exist_ok=True)
    save_file(tensors, os.path.join(a.out, SALTQ_BASE_FILENAME))
    mc = dict(mc); mc["salient_init"] = "gptq_latent"
    mc["derived_from"] = {"codes_base": os.path.abspath(a.codes_base), "fp16_base": os.path.abspath(a.fp16_base)}
    torch.save(mc, os.path.join(a.out, SALTQ_META_FILENAME))
    for f in os.listdir(a.codes_base):
        if f.startswith("tokenizer"):
            shutil.copy(os.path.join(a.codes_base, f), os.path.join(a.out, f))
    print(f"[latent-base] wrote {a.out}: {n_kept / n_tot * 100:.2f}% of {n_tot / 1e6:.1f}M salient starts keep their "
          f"fp16 value, {100 - n_kept / n_tot * 100:.2f}% sit at the GPTQ cell centre; every code unchanged (asserted).")

if __name__ == "__main__":
    main()
