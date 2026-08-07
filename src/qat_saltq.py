"""
qat_saltq.py — SALT-Q: Saliency-Allocated Low-bit Trainability.

The method is defined by how it ALLOCATES TRAINABLE DEGREES OF FREEDOM across a weight matrix,
according to saliency. After the shared offline permutation has moved every segment's salient
input channels into the leading `group_k` columns (see `permute_common`), each target linear is
split into two segments that are trained in COMPLETELY DIFFERENT ways:

    columns [0 : group_k)   ~1-2%    FULL weight-space freedom
        The weights themselves are nn.Parameters, trained with LSQ fake-quant (STE). At the end
        of training they already sit on the quantization grid, so "merging" is just rounding.
        This segment carries the high-rank part of the task adaptation.

    columns [group_k : in)  ~98-99%  AFFINE-ONLY freedom
        The integer codes q are FROZEN (GPTQ-initialized). Only the per-(output row, quant group)
        zero-point z is trained by default; the scale s stays frozen (train_scale=True restores it
        as an ablation). The deployed weight is (q - z) * s, and s/z are written straight into the
        metadata slots a normal GPTQ checkpoint already has.
        This segment carries the intra-group shift part of the task adaptation.

    the codes themselves                ZERO freedom.

Training z rather than (s, z) is not a shortcut, it is what makes the method cheap. dL/dz is a
UNIFORM sum over each group, so it collapses into a pooled input and costs 2N/g; dL/ds carries a
per-element (q-z) factor that folds into neither operand and forces the full [out, in] weight-
gradient GEMM at 2N no matter how few scale parameters exist. With s frozen, q*s is also constant,
so the forward is a frozen GEMM plus a pooled correction and no weight is ever reconstructed. See
the SALTQLinear docstring for the full ledger.

Consequences — this is the whole point:

  * There is NO BF16 object anywhere in the pipeline that would need to be merged. No merge step,
    no re-quantization damage, inference is uniformly low-bit. `export_saltq` asserts that the
    deployed weight equals the training-time effective weight EXACTLY (max|Δ| == 0), which is the
    machine-checkable form of that claim.
  * The base never passes through NF4. Unlike the QLoRA-based methods here (where 2/3-bit training
    runs on an NF4 base and applies the real bit-width only inside the fakequant), SALT-Q trains
    on the actual INT-b grid it deploys on. No double quantization.
  * o_proj, which has no contiguous salient slice (per-head structure forbids cross-head
    permutation, so `group_k_for_module_name` returns 0 for it), becomes trainable for the first
    time: it is pure frozen-codes + trainable (s, z).

Positioning:
  * QA-LoRA is a DEGENERATE SPECIAL CASE of the non-salient segment. Its group-pooled adapter
    produces a delta that is constant within each input group, which folds exactly into the affine
    zero-point — i.e. it trains a rank-r constrained parameterization of z alone. SALT-Q trains z
    directly per (row, group), full rank. The resemblance is not superficial: the z-only forward
    here IS a pooled-input GEMM, the same structure QA-LoRA's adapter has, minus the rank bottleneck.
  * LR-QAT fake-quantizes every weight each step and still needs a BF16 low-rank term. SALT-Q
    fake-quantizes only the 1-2% salient slice, and the non-salient segment needs no fakequant and
    no reconstruction at all.

Pipeline (driven by scripts/train.py):

    1. permute_common.build_permuted_fp16_checkpoint()   -> permuted fp16 base + perm_meta
    2. build_saltq_base()                                -> frozen codes + (s0, z0) + salient init
         (one call to gptq.gptq_quantize_model_sequential in the permuted basis)
    3. build_saltq_model()                               -> HF model with SALTQLinear layers
    4. SALTQ handler                                     -> boundary gathers + meta
    5. export_saltq()                                    -> dense deployed model (+ perm meta)

Layout on disk (`saltq_base/`):
    saltq_base.safetensors   {name}.codes int8 [out, in_n] | {name}.s_n/.z_n | {name}.w_s/.s_s/.z_s
    saltq_meta.pt            perm_meta + q_bits/group_size/symmetric + per-layer shapes/group_k
"""

import gc
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .qat_base import (
    QATHandler,
    grad_scale,
    groupwise_lsq_asym_fakequant,
    groupwise_lsq_symmetric_fakequant,
    init_lsq_scale_sym,
    init_lsq_scale_zp_asym,
    lsq_quantize_export_asym,
    lsq_quantize_export_sym,
    round_ste,
)
from .gptq import gptq_quantize_model_sequential
from .permute_common import (
    PERM_META_FILENAME,
    format_fakequant_param_stats,
    group_k_for_module_name,
    register_boundary_gathers_from_meta,
)

SALTQ_BASE_FILENAME = "saltq_base.safetensors"
SALTQ_META_FILENAME = "saltq_meta.pt"
SALTQ_TRAINABLE_FILENAME = "saltq_trainable.safetensors"
SALTQ_CKPT_META_FILENAME = "saltq_ckpt_meta.pt"

# Parameter-name fragments used by the optimizer grouping in src/trainer.py.
# Optimizer grouping is by the UNITS a parameter lives in, not by whether it is a "quant param".
# Mixing them was a real bug: lsq_w_zp and saltq_z are measured in quantization LEVELS (meaningful
# step 1.0) while lsq_w_scale and saltq_s are in weight units (meaningful step ~1e-2). Sharing one
# lr left both zero-points at 0.0000% change over a full training run.
SALIENT_WEIGHT_PARAM = "weight_salient"
SCALE_PARAM_FRAGMENTS = ("lsq_w_scale", "saltq_s")
ZEROPOINT_PARAM_FRAGMENTS = ("lsq_w_zp", "saltq_z")
QUANT_PARAM_FRAGMENTS = SCALE_PARAM_FRAGMENTS + ZEROPOINT_PARAM_FRAGMENTS


def _lsq_levels(q_bits: int, symmetric: bool) -> Tuple[int, int]:
    """LSQ grid bounds. Must match qat_base._lsq_qn_qp — the export path uses the same."""
    if symmetric:
        return -(2 ** (q_bits - 1)), 2 ** (q_bits - 1) - 1
    return 0, 2 ** q_bits - 1


# ============================================================================
# Part 1 — SALT-Q base builder (GPTQ initialization, run ONCE on rank 0)
# ============================================================================

@torch.no_grad()
def build_saltq_base(
    permuted_base_dir: str,
    perm_meta: dict,
    tokenizer,
    calibration_dataloader: DataLoader,
    target_terminals: Sequence[str],
    *,
    group_size: int,
    q_bits: int,
    symmetric: bool,
    save_dir: str,
    device=None,
    percdamp: float = 0.01,
    blocksize: int = 128,
    nsamples: int = 128,
    dtype: torch.dtype = torch.float16,
) -> dict:
    """
    Turn a permuted fp16 base into SALT-Q's starting point.

    For every target linear (in the permuted basis, boundary gathers active):
      * salient columns [0:group_k)  -> kept as fp32 WEIGHTS (the trainable init) plus a
        `current_minmax` LSQ scale[, zp]. They are NOT frozen and NOT stored as codes.
      * non-salient columns [group_k:) -> GPTQ integer codes (FROZEN forever) + the GPTQ grid
        (s, z) as the trainable init.

    The GPTQ sweep is the standard one from `gptq.gptq_quantize_model_sequential`: sequential
    across decoder layers (each layer sees the already-quantized previous layers' activations),
    salient slice pinned to the canonical grid, non-salient block solved as an INDEPENDENT OBS
    problem on H[group_k:, group_k:]. That is exactly the initialization we want, because at
    step 0 SALT-Q's salient segment reproduces the canonical grid too (asym LSQ+ at current_minmax
    IS the canonical affine grid), so the model starts numerically identical to a GPTQ export.

    Returns the meta dict (also written to save_dir/saltq_meta.pt).
    """
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_terminals = list(target_terminals)
    max_group_k = int(perm_meta["group_k"])

    print(f"[SALT-Q] Loading permuted base {permuted_base_dir} in {dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(
        permuted_base_dir, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    # --- capture the salient slices BEFORE GPTQ overwrites the weights in place ---
    salient_fp: Dict[str, torch.Tensor] = {}
    shapes: Dict[str, Tuple[int, int, int]] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if name.split(".")[-1] not in target_terminals:
            continue
        gk = group_k_for_module_name(name, perm_meta=perm_meta, default_group_k=max_group_k)
        out_f, in_f = int(mod.weight.shape[0]), int(mod.weight.shape[1])
        if in_f % group_size != 0:
            raise ValueError(
                f"[SALT-Q] {name}: in_features={in_f} not divisible by group_size={group_size}"
            )
        if gk >= in_f:
            raise ValueError(f"[SALT-Q] {name}: group_k={gk} must be < in_features={in_f}")
        shapes[name] = (out_f, in_f, int(gk))
        if gk > 0:
            salient_fp[name] = mod.weight.data[:, :gk].detach().float().cpu().clone()
    print(f"[SALT-Q] Captured salient init for {len(salient_fp)} projections "
          f"({sum(v.numel() for v in salient_fp.values()) / 1e6:.1f}M trainable weights)")

    # --- GPTQ the permuted base (this is the frozen-code initialization) ---
    quantized = gptq_quantize_model_sequential(
        model,
        calibration_dataloader,
        target_terminals,
        perm_group_k=max_group_k,
        group_size=group_size,
        q_bits=q_bits,
        symmetric=symmetric,
        device=device,
        perm_meta=perm_meta,
        percdamp=percdamp,
        blocksize=blocksize,
        nsamples=nsamples,
        awq_scales=None,      # SALT-Q trains the salient weights directly; AWQ-S is redundant
        lsq_scales=None,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- split into the two segments and persist ---
    os.makedirs(save_dir, exist_ok=True)
    tensors: Dict[str, torch.Tensor] = {}
    layers_meta: Dict[str, dict] = {}
    n_codes = 0
    n_salient = 0
    n_qparams = 0

    # Drain `quantized` as we go: gptq_quantize_model_sequential hands back the integer levels as
    # fp32 [out, in] tensors, which for a 7B model is ~27 GB of host RAM. Popping each entry right
    # after narrowing it to int8 keeps the peak from stacking the fp32 originals on top of the
    # int8 copies.
    for name in list(quantized.keys()):
        W_int, scale, zp = quantized.pop(name)
        out_f, in_f, gk = shapes[name]
        n_sal_g = gk // group_size

        codes = W_int[:, gk:].round().to(torch.int8).contiguous()
        tensors[f"{name}.codes"] = codes
        tensors[f"{name}.s_n"] = scale[:, n_sal_g:].float().contiguous()
        tensors[f"{name}.z_n"] = zp[:, n_sal_g:].float().contiguous()
        n_codes += codes.numel()
        n_qparams += 2 * tensors[f"{name}.s_n"].numel()

        if gk > 0:
            W_S = salient_fp[name]
            tensors[f"{name}.w_s"] = W_S.contiguous()
            if symmetric:
                s0 = init_lsq_scale_sym(W_S, group_size, q_bits)
                tensors[f"{name}.s_s"] = s0.float().contiguous()
            else:
                s0, z0 = init_lsq_scale_zp_asym(W_S, group_size, q_bits)
                tensors[f"{name}.s_s"] = s0.float().contiguous()
                tensors[f"{name}.z_s"] = z0.float().contiguous()
            n_salient += W_S.numel()
            n_qparams += (1 if symmetric else 2) * s0.numel()

        layers_meta[name] = {
            "out_features": out_f, "in_features": in_f, "group_k": gk,
            "n_nonsalient": in_f - gk,
        }
        salient_fp.pop(name, None)
        del W_int, scale, zp

    gc.collect()
    save_file(tensors, os.path.join(save_dir, SALTQ_BASE_FILENAME))

    meta = {
        "method": "saltq",
        "permuted_base_dir": os.path.abspath(permuted_base_dir),
        "q_bits": int(q_bits),
        "group_size": int(group_size),
        "symmetric": bool(symmetric),
        "target_terminals": target_terminals,
        "layers": layers_meta,
        "perm_meta": perm_meta,
        "param_counts": {
            "frozen_codes": int(n_codes),
            "trainable_salient_weights": int(n_salient),
            "trainable_qparams": int(n_qparams),
        },
    }
    torch.save(meta, os.path.join(save_dir, SALTQ_META_FILENAME))

    tokenizer.save_pretrained(save_dir)
    print(
        f"[SALT-Q] base saved -> {save_dir}\n"
        f"[SALT-Q]   frozen codes:              {n_codes / 1e6:9.1f}M  (int8, zero freedom)\n"
        f"[SALT-Q]   trainable salient weights: {n_salient / 1e6:9.1f}M  (full weight freedom)\n"
        f"[SALT-Q]   trainable (s, z):          {n_qparams / 1e6:9.1f}M  (affine freedom)"
    )
    return meta


# ============================================================================
# Part 2 — SALTQLinear: the two-segment layer
# ============================================================================

class SALTQLinear(nn.Module):
    """
    A linear layer whose input columns are split by saliency into two differently-trained segments.

        y = x @ [ LSQ_fakequant(W_S ; s_S, z_S) | (q_N - round(z_N)) * s_N ]^T   (+ bias)
              \\__________ trainable weights ___/   \\__ frozen codes, trainable (s, z) __/

    `group_k == 0` (o_proj) degenerates to a pure frozen-codes layer.

    ------------------------------------------------------------------------------------------
    Why the default is z-only (train_scale=False), and why that makes training CHEAP
    ------------------------------------------------------------------------------------------
    Write G_W = dL/dW = gy^T x. The three trainable groups cost wildly different amounts:

      dL/dW_S   = gy^T x_S                       only the ~1-2% salient columns  -> 2N*k, free
      dL/dz_o,g = -s_o,g * SUM_{j in g} G_W[o,j]
                = -s_o,g * SUM_t gy[t,o] * (SUM_{j in g} x[t,j])
                                                 the inner sum is UNIFORM over the group, so it
                                                 collapses into a pooled input BEFORE the GEMM:
                                                 [out,T] x [T, in/g] -> 2N/g, ~1.5% at g=64, free
      dL/ds_o,g = SUM_{j in g} (q-z)[o,j] * G_W[o,j]
                                                 the per-element factor (q-z)[o,j] depends on BOTH
                                                 o and j, so it folds into neither x nor gy. This
                                                 is a genuine three-way contraction and forces the
                                                 full [out,in] weight-gradient GEMM -> 2N, and 2N
                                                 REGARDLESS of how few scale parameters there are.

    So the ledger is 6N + 2N*k + 2N/g + 2N*[train_scale], and only the scale term is irreducible.
    Training z alone therefore costs the same as QLoRA/PSQAT while strictly containing QA-LoRA's
    adaptation space: QA-LoRA's group-pooled adapter IS a rank-r constrained parameterization of
    exactly this delta-z, and here it is full rank.

    Dropping s from training also removes the forward-side cost. With s frozen, q*s is CONSTANT,
    so it is precomputed ONCE into `wq` and the forward becomes

        y_N = x @ wq^T  -  pool_g(x) @ (z*s)^T

    a frozen GEMM plus a [T, in/g] x [in/g, out] correction. No [out, in] tensor is ever
    reconstructed, in either direction — training touches the same amount of memory traffic as a
    plain quantized forward. Ordinary autograd on this expression produces exactly the cheap
    gradients derived above; no custom autograd.Function is needed.

    `train_scale=True` restores the trainable scale for ablations and falls back to the
    reconstruct-then-GEMM path, paying the full 2N.

    ------------------------------------------------------------------------------------------
    Storage
    ------------------------------------------------------------------------------------------
    `wq` is a non-persistent buffer (rebuilt from the frozen base at load) and the integer codes
    live on the HOST: only export and verification need them, and keeping them off the accelerator
    saves several GB. Neither enters `state_dict()`, so a checkpoint holds only what changes.

    `effective_weight()` / `deployed_weight()` remain the fp32 REFERENCE definitions built from
    (codes, s, z) — the merge-free guarantee is a statement about those, and the fast forward path
    is an algebraic rearrangement of the same quantity evaluated in the compute dtype.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_k: int,
        group_size: int,
        q_bits: int,
        symmetric: bool,
        *,
        codes: torch.Tensor,
        s_n: torch.Tensor,
        z_n: torch.Tensor,
        w_s: Optional[torch.Tensor] = None,
        s_s: Optional[torch.Tensor] = None,
        z_s: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        param_dtype: torch.dtype = torch.float32,
        train_scale: bool = False,
        continuous_z: bool = True,
        wq_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        assert group_k % group_size == 0, \
            f"group_k={group_k} must be a multiple of group_size={group_size}"
        assert in_features % group_size == 0, \
            f"in_features={in_features} must be a multiple of group_size={group_size}"

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_k = int(group_k)
        self.group_size = int(group_size)
        self.q_bits = int(q_bits)
        self.symmetric = bool(symmetric)
        self.train_scale = bool(train_scale)
        self.continuous_z = bool(continuous_z)
        self.n_nonsalient = self.in_features - self.group_k
        self.ng = self.in_features // self.group_size
        self.n_nonsal_g = self.n_nonsalient // self.group_size
        self.n_sal_g = self.group_k // self.group_size
        self.Qn, self.Qp = _lsq_levels(self.q_bits, self.symmetric)

        # LSQ-style gradient damping so the (s, z) group's gradients are on the same scale as
        # the weight gradients (identical factors to qat_base.groupwise_lsq_*).
        self._s_gfactor = 1.0 / math.sqrt(self.group_size * max(self.Qp, 1))
        self._z_gfactor = 1.0 / math.sqrt(self.group_size)

        # ---- non-salient segment ----
        # Codes stay on the host: export and verification read them, the training forward does not.
        # A plain attribute (not a buffer) so .to(device) leaves them where they are.
        self._codes_cpu = codes.to(torch.int8).contiguous().cpu()
        if self.train_scale:
            self.saltq_s = nn.Parameter(s_n.float().contiguous())
        else:
            self.register_buffer("saltq_s", s_n.float().contiguous(), persistent=False)
        if not self.symmetric:
            self.saltq_z = nn.Parameter(z_n.float().contiguous())

        # Precompute the frozen part of the deployed weight. Only valid while s is frozen.
        # Salient columns are left at zero: their contribution comes from weight_salient.
        if not self.train_scale:
            wq = torch.zeros(self.out_features, self.in_features, dtype=wq_dtype)
            q = codes.float().view(self.out_features, self.n_nonsal_g, self.group_size)
            wq[:, self.group_k:] = (
                (q * s_n.float().unsqueeze(-1))
                .reshape(self.out_features, self.n_nonsalient).to(wq_dtype)
            )
            self.register_buffer("wq", wq, persistent=False)
            del q

        # ---- salient segment: trainable weights + trainable LSQ grid ----
        if self.group_k > 0:
            assert w_s is not None and s_s is not None
            self.weight_salient = nn.Parameter(w_s.to(param_dtype).contiguous())
            self.lsq_w_scale = nn.Parameter(s_s.float().contiguous())
            if not self.symmetric:
                assert z_s is not None
                self.lsq_w_zp = nn.Parameter(z_s.float().contiguous())

        if bias is not None:
            self.register_buffer("bias", bias.clone(), persistent=True)
        else:
            self.bias = None

    # ------------------------------------------------------------------ pieces

    def _salient_effective(self) -> torch.Tensor:
        """LSQ fake-quantized salient weights, fp32 [out, group_k]. STE through the rounding."""
        W = self.weight_salient.float()
        if self.symmetric:
            return groupwise_lsq_symmetric_fakequant(
                W, self.lsq_w_scale.float(), self.group_size, self.q_bits
            )
        return groupwise_lsq_asym_fakequant(
            W, self.lsq_w_scale.float(), self.lsq_w_zp.float(), self.group_size, self.q_bits
        )

    def _z_eff(self) -> torch.Tensor:
        """
        The zero-point actually used, [out, n_nonsal_g].

        continuous_z=True (default): a FRACTIONAL zero-point, no rounding.

        An integer-constrained z cannot be trained. z lives in units of quantization LEVELS, where
        the meaningful step is 1.0, so an STE round puts a dead zone in front of it: gradients flow
        but the effective weight does not move until z crosses a half level. Measured on the first
        INT3 g64 run, |dz| reached 4.4e-3 against a 0.5 threshold and round(z) changed for 0 of
        1.63M parameters — the entire non-salient segment contributed exactly nothing.

        A fractional zero-point is also what QA-LoRA actually deploys: its group-pooled delta c
        folds in as z_eff = z - c/s, which is not an integer. Integer z is therefore STRICTLY
        WEAKER than the baseline it is meant to generalize. The cost is that the zero-point ships
        as fp16 metadata rather than packed int4 (+16 bits per group, ~0.25 bit/weight at g=64).

        continuous_z=False keeps the integer grid for the ablation; it needs a much larger lr to
        move at all, and adapts only in whole quantization steps.
        """
        z = self.saltq_z.float()
        if not self.continuous_z:
            z = round_ste(grad_scale(z, self._z_gfactor))
        return z.clamp(self.Qn, self.Qp)

    def _s_eff(self) -> torch.Tensor:
        s = self.saltq_s.float().clamp(min=1e-8)
        return grad_scale(s, self._s_gfactor) if self.train_scale else s

    def _nonsalient_effective(self, codes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reconstruct (q - z) * s from the FROZEN codes, fp32 [out, n_nonsalient].

        This is the REFERENCE definition (and the training path when train_scale=True). The
        z-only forward never calls it — see the class docstring.
        """
        q = (self._codes_cpu if codes is None else codes)
        s = self._s_eff()
        q = q.to(device=s.device, dtype=torch.float32).view(
            self.out_features, self.n_nonsal_g, self.group_size)
        s = s.unsqueeze(-1)
        if self.symmetric:
            W = q * s
        else:
            W = (q - self._z_eff().unsqueeze(-1)) * s
        return W.reshape(self.out_features, self.n_nonsalient)

    def effective_weight(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        fp32 reference weight: exactly what the deployed checkpoint reconstructs to.

        Built on the device the PARAMETERS live on, pulling the codes over from the host, so it is
        memory-hungry by design — it exists for verification and export, not for the hot path.
        """
        parts: List[torch.Tensor] = []
        if self.group_k > 0:
            parts.append(self._salient_effective())
        parts.append(self._nonsalient_effective())
        W = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        return W if dtype is None else W.to(dtype)

    # ---------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None

        if self.train_scale:
            # Ablation path: s is trainable, so q*s is not constant and the weight has to be
            # rebuilt. This is the configuration that costs the extra 2N.
            return F.linear(x, self.effective_weight(dtype=x.dtype), bias)

        # Frozen main GEMM. wq holds q*s with the salient columns zeroed.
        out = F.linear(x, self.wq, bias)

        # Zero-point correction. (q - z)*s = q*s - z*s and z*s is constant within a group, so the
        # whole correction is one pooled-input GEMM instead of a full [out, in] reconstruction.
        if not self.symmetric:
            zs = (self._z_eff() * self._s_eff()).to(x.dtype)            # [out, n_nonsal_g]
            xp = x.reshape(*x.shape[:-1], self.ng, self.group_size).sum(-1)
            out = out - F.linear(xp[..., self.n_sal_g:], zs)

        if self.group_k > 0:
            out = out + F.linear(x[..., :self.group_k],
                                 self._salient_effective().to(x.dtype))
        return out

    # -------------------------------------------------------------- deployment

    @torch.no_grad()
    def deployed_tensors(self):
        """
        The exact contents of a deployed INT-b checkpoint for this layer:

            codes  [out, in]     int (salient part = rounded trained weights, non-salient = frozen)
            scale  [out, ng]
            zp     [out, ng]     integer-valued (zeros for the symmetric grid)

        `group_dequantize`-style reconstruction of these equals `effective_weight()` exactly.
        """
        scales: List[torch.Tensor] = []
        zeros: List[torch.Tensor] = []
        codes: List[torch.Tensor] = []

        if self.group_k > 0:
            W = self.weight_salient.float()
            s_s = self.lsq_w_scale.float()
            if self.symmetric:
                q_s = lsq_quantize_export_sym(W, s_s, self.group_size, self.q_bits)
                z_s = torch.zeros_like(s_s)
            else:
                q_s, z_s = lsq_quantize_export_asym(
                    W, s_s, self.lsq_w_zp.float(), self.group_size, self.q_bits
                )
            codes.append(q_s)
            scales.append(s_s)
            zeros.append(z_s)

        s_n = self.saltq_s.float().clamp(min=1e-8)
        if self.symmetric:
            z_n = torch.zeros_like(s_n)
        else:
            z_n = self.saltq_z.float()
            if not self.continuous_z:
                z_n = z_n.round()
            z_n = z_n.clamp(self.Qn, self.Qp)
        codes.append(self._codes_cpu.to(device=s_n.device, dtype=torch.float32))
        scales.append(s_n)
        zeros.append(z_n)

        return (
            torch.cat(codes, dim=1),
            torch.cat(scales, dim=1),
            torch.cat(zeros, dim=1),
        )

    @torch.no_grad()
    def deployed_weight(self) -> torch.Tensor:
        """Dense fp32 reconstruction of the deployed checkpoint. MUST equal effective_weight()."""
        q, s, z = self.deployed_tensors()
        ng = s.shape[1]
        W = q.view(self.out_features, ng, self.group_size)
        W = (W - z.unsqueeze(-1)) * s.unsqueeze(-1)
        return W.reshape(self.out_features, self.in_features)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, group_k={self.group_k}, "
            f"group_size={self.group_size}, bits={self.q_bits}, "
            f"symmetric={self.symmetric}"
        )


# ============================================================================
# Part 3 — Model assembly
# ============================================================================

def _split_parent(name: str) -> Tuple[str, str]:
    parent, _, terminal = name.rpartition(".")
    return parent, terminal


def build_saltq_model(
    saltq_base_dir: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    param_dtype: torch.dtype = torch.float32,
    gradient_checkpointing: bool = True,
    train_layernorms: bool = False,
    train_scale: bool = False,
    continuous_z: bool = True,
    device=None,
):
    """
    Load the permuted fp16 base and swap every target linear for a SALTQLinear.

    Replacement is STREAMING: each fp16 weight is dropped as soon as its SALTQLinear is built, so
    the peak host memory is ~max(fp16 base, codes) rather than their sum. Everything that is not
    a SALTQLinear parameter is frozen (optionally except the LayerNorms).

    Returns (model, meta).
    """
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    meta = torch.load(
        os.path.join(saltq_base_dir, SALTQ_META_FILENAME), map_location="cpu", weights_only=False
    )
    group_size = int(meta["group_size"])
    q_bits = int(meta["q_bits"])
    symmetric = bool(meta["symmetric"])

    print(f"[SALT-Q] Loading permuted base {meta['permuted_base_dir']} ...")
    model = AutoModelForCausalLM.from_pretrained(
        meta["permuted_base_dir"], torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    path = os.path.join(saltq_base_dir, SALTQ_BASE_FILENAME)
    n_replaced = 0
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for name, info in meta["layers"].items():
            parent_name, terminal = _split_parent(name)
            parent = model.get_submodule(parent_name)
            old = getattr(parent, terminal)
            bias = None
            if getattr(old, "bias", None) is not None:
                bias = old.bias.data.detach().clone()

            layer = SALTQLinear(
                in_features=int(info["in_features"]),
                out_features=int(info["out_features"]),
                group_k=int(info["group_k"]),
                group_size=group_size,
                q_bits=q_bits,
                symmetric=symmetric,
                codes=f.get_tensor(f"{name}.codes"),
                s_n=f.get_tensor(f"{name}.s_n"),
                z_n=f.get_tensor(f"{name}.z_n"),
                w_s=f.get_tensor(f"{name}.w_s") if f"{name}.w_s" in keys else None,
                s_s=f.get_tensor(f"{name}.s_s") if f"{name}.s_s" in keys else None,
                z_s=f.get_tensor(f"{name}.z_s") if f"{name}.z_s" in keys else None,
                bias=bias,
                param_dtype=param_dtype,
                train_scale=train_scale,
                continuous_z=continuous_z,
                wq_dtype=dtype,
            )
            setattr(parent, terminal, layer)
            del old
            n_replaced += 1
            if n_replaced % 64 == 0:
                gc.collect()
    gc.collect()

    # ---- freedom allocation: everything frozen except the SALT-Q parameters ----
    model.requires_grad_(False)
    n_train = 0
    for mod in model.modules():
        if isinstance(mod, SALTQLinear):
            for p in mod.parameters():
                p.requires_grad_(True)
                n_train += p.numel()
    if train_layernorms:
        for name, p in model.named_parameters():
            if "layernorm" in name.lower() or name.endswith("model.norm.weight"):
                p.requires_grad_(True)
                n_train += p.numel()

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    total = sum(p.numel() for p in model.parameters())
    n_codes = meta["param_counts"]["frozen_codes"]
    mode = "s+z (ablation, pays the full 2N weight-gradient GEMM)" if train_scale \
        else "z-only (default; forward is a frozen GEMM + pooled correction)"
    mode += ", continuous z (fractional zero-point)" if continuous_z else ", INTEGER z (ablation)"
    print(
        f"[SALT-Q] Replaced {n_replaced} linears; trainable {n_train / 1e6:.1f}M params "
        f"({100.0 * n_train / max(total, 1):.2f}% of the fp-parameter tree)\n"
        f"[SALT-Q] affine training mode: {mode}\n"
        f"[SALT-Q] frozen: {n_codes / 1e6:.1f}M codes (host, int8, export/verify only)"
        + ("" if train_scale else
           f" -> precomputed q*s on device as {dtype} "
           f"({n_codes * torch.finfo(dtype).bits / 8 / 1e9:.1f} GB)")
    )
    if device is not None:
        model.to(device)
    return model, meta


def saltq_layers(model: nn.Module) -> Dict[str, SALTQLinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, SALTQLinear)}


# ============================================================================
# Part 4 — Trainable-state checkpointing (codes are never written)
# ============================================================================

def saltq_trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        n: p.detach().cpu().contiguous()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def save_saltq_trainable(model: nn.Module, output_dir: str, saltq_base_dir: Optional[str] = None,
                         learning_rates: Optional[dict] = None):
    """
    Persist ONLY the trainable tensors (salient weights + all (s, z)) plus a pointer to the frozen
    base. A full state_dict would contain multiple GB of int8 codes that never change.

    `learning_rates` records the three RESOLVED per-group lrs. They belong in the artifact rather
    than being re-read from the yaml at collection time: the yaml holds a per-bit table plus CLI
    overrides, so it is not a record of what any particular run used.
    """
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)
    sd = saltq_trainable_state_dict(model)
    save_file(sd, os.path.join(output_dir, SALTQ_TRAINABLE_FILENAME))
    meta = getattr(model, "_saltq_meta", None)
    torch.save(
        {
            "saltq_base_dir": os.path.abspath(saltq_base_dir) if saltq_base_dir
            else (meta or {}).get("saltq_base_dir"),
            "num_tensors": len(sd),
            "num_params": int(sum(v.numel() for v in sd.values())),
            "learning_rates": dict(learning_rates or {}),
        },
        os.path.join(output_dir, SALTQ_CKPT_META_FILENAME),
    )
    print(f"[SALT-Q] Saved {len(sd)} trainable tensors "
          f"({sum(v.numel() for v in sd.values()) / 1e6:.1f}M params) to {output_dir}")


def load_saltq_trainable(model: nn.Module, checkpoint_dir: str) -> int:
    """Overlay a trained checkpoint onto a freshly built SALT-Q model. Returns tensors loaded."""
    from safetensors.torch import load_file

    path = os.path.join(checkpoint_dir, SALTQ_TRAINABLE_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[SALT-Q] {path} not found — a SALT-Q checkpoint stores only the trainable tensors; "
            f"the frozen codes live in the saltq_base dir referenced by "
            f"{SALTQ_CKPT_META_FILENAME}."
        )
    sd = load_file(path)
    own = dict(model.named_parameters())
    missing = [k for k in own if own[k].requires_grad and k not in sd]
    unexpected = [k for k in sd if k not in own]
    if missing:
        raise RuntimeError(f"[SALT-Q] checkpoint is missing {len(missing)} trainable tensors, "
                           f"e.g. {missing[:3]}")
    if unexpected:
        print(f"[SALT-Q] WARNING: ignoring {len(unexpected)} unexpected checkpoint tensors")
    with torch.no_grad():
        for k, v in sd.items():
            if k in own:
                own[k].copy_(v.to(own[k].dtype).to(own[k].device))
    print(f"[SALT-Q] Loaded {len(sd)} trainable tensors from {path}")
    return len(sd)


def saltq_base_dir_from_checkpoint(checkpoint_dir: str) -> Optional[str]:
    p = os.path.join(checkpoint_dir, SALTQ_CKPT_META_FILENAME)
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location="cpu", weights_only=False).get("saltq_base_dir")


# ============================================================================
# Part 5 — QAT handler
# ============================================================================

class SALTQ(QATHandler):
    """
    SALT-Q handler.

    The heavy lifting (permute, GPTQ init, module replacement) happens before this point, in
    scripts/train.py. All that remains is the one thing that CANNOT be folded into weights:
    the runtime residual reorder at each segment boundary.
    """

    def __init__(self):
        self.boundary_hooks: List = []
        self.meta: Optional[dict] = None
        self.saltq_base_dir: Optional[str] = None

    def prepare_model(self, model: nn.Module, cfg: dict, saltq_meta: Optional[dict] = None,
                      saltq_base_dir: Optional[str] = None, **kwargs) -> nn.Module:
        assert saltq_meta is not None, (
            "SALT-Q prepare_model requires saltq_meta from build_saltq_base(); the base is built "
            "in scripts/train.py before the model is assembled."
        )
        self.meta = saltq_meta
        self.saltq_base_dir = saltq_base_dir
        perm_meta = saltq_meta["perm_meta"]

        # The skip connection has no weight carrier, so P_k -> P_{k+1} must happen at runtime.
        # Required at training AND at inference (the exported model carries the same meta).
        self.boundary_hooks = register_boundary_gathers_from_meta(model, perm_meta)

        model._saltq_meta = {**saltq_meta, "saltq_base_dir": saltq_base_dir}

        layers = saltq_layers(model)
        gks = sorted({m.group_k for m in layers.values()})
        counts = saltq_meta["param_counts"]
        print(
            f"[SALT-Q] prepare_model done: {len(layers)} SALT-Q linears, "
            f"num_runtime_permutes={len(self.boundary_hooks)}, "
            f"group_k values={gks}, group_size={saltq_meta['group_size']}, "
            f"bits={saltq_meta['q_bits']}, symmetric={saltq_meta['symmetric']}"
        )
        print(
            f"[SALT-Q] freedom allocation: "
            f"{counts['trainable_salient_weights'] / 1e6:.1f}M full-freedom salient weights + "
            f"{counts['trainable_qparams'] / 1e6:.1f}M affine (s,z) + "
            f"{counts['frozen_codes'] / 1e6:.1f}M zero-freedom codes"
        )
        if perm_meta.get("fakequant_param_stats"):
            print("[SALT-Q] salient coverage: "
                  f"{format_fakequant_param_stats(perm_meta['fakequant_param_stats'])}")
        return model

    def on_train_begin(self, model: nn.Module):
        pass

    def on_step_end(self, model: nn.Module, step: int):
        pass

    def on_train_end(self, model: nn.Module, output_dir: Optional[str] = None):
        pass


# ============================================================================
# Part 6 — Export (no merge: the trained weights ARE the deployed weights)
# ============================================================================

@torch.no_grad()
def verify_saltq_deploy_equivalence(model: nn.Module, max_layers: int = 8) -> float:
    """
    Assert that the deployed checkpoint reproduces the training-time effective weight EXACTLY.

    This is the machine-checkable form of "merge-free": for the salient segment the export
    rounding is literally the same function the training STE applied, and for the non-salient
    segment the codes are untouched and (s, z) are carried over verbatim. The tolerance is 0,
    not an epsilon.
    """
    worst = 0.0
    checked = 0
    for name, layer in saltq_layers(model).items():
        if checked >= max_layers:
            break
        err = (layer.deployed_weight() - layer.effective_weight()).abs().max().item()
        worst = max(worst, err)
        status = "OK" if err == 0.0 else "MISMATCH"
        print(f"  [SALT-Q deploy] {name}: max|Δ|={err:.3e} [{status}]")
        checked += 1
    if worst != 0.0:
        raise RuntimeError(
            f"[SALT-Q] deployed weight != training weight (max|Δ|={worst:.3e}). The merge-free "
            f"guarantee is broken — check that the export quantizer and the training fakequant "
            f"use the same LSQ grid."
        )
    print(f"[SALT-Q] deploy equivalence verified on {checked} layers (max|Δ| = 0, exact).")
    return worst


def _resolve_export_dtype(name: str) -> torch.dtype:
    key = name.lower()
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("fp32", "float32", "float"):
        return torch.float32
    return torch.bfloat16


@torch.no_grad()
def materialize_dense_model(
    model: nn.Module, dtype: torch.dtype = torch.float16,
) -> Tuple[nn.Module, float]:
    """
    Replace every SALTQLinear with a plain nn.Linear holding the deployed weight.

    Returns (model, max_cast_error). The cast is the ONE lossy step in the whole SALT-Q pipeline,
    and it is an artifact of shipping a DENSE checkpoint for evaluation rather than a packed INT-b
    one: the deployed values (q - z) * s are exact in fp32 but get rounded to the storage dtype.
    A real packed export would carry the integers and the fp16 scale and have no such step. The
    error is returned (and printed by export_saltq) so it is never silently assumed to be zero.
    """
    worst = 0.0
    for name, layer in list(saltq_layers(model).items()):
        parent_name, terminal = _split_parent(name)
        parent = model.get_submodule(parent_name)
        W = layer.deployed_weight()
        W_cast = W.to(dtype)
        worst = max(worst, (W_cast.float() - W).abs().max().item())
        dense = nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
        dense.weight.data = W_cast
        if layer.bias is not None:
            dense.bias.data = layer.bias.to(dtype)
        setattr(parent, terminal, dense)
        del W, W_cast
    gc.collect()
    return model, worst


@torch.no_grad()
def export_saltq(
    cfg: dict,
    tokenizer,
    saltq_base_dir: str,
    checkpoint_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    model: Optional[nn.Module] = None,
    dequant_dtype: str = "float16",
) -> str:
    """
    Export a SALT-Q model for evaluation.

    There is no merge step and no re-quantization: the deployed weight is read straight out of the
    trained parameters. What is written is a DENSE checkpoint whose weights are exactly the
    (q - z) * s values a packed INT-b checkpoint would dequantize to — the same convention the rest
    of this repo uses for evaluation (`--export_dequant`), so lm-eval can load it directly. The
    boundary-gather metadata rides along under the usual filename so the eval glue registers it.

    Either pass a live `model` (in-training export) or a `checkpoint_dir` (export-only).
    """
    from .export import save_dequantized_model, save_sqat_permute_meta

    q_bits = cfg["model"]["quant_bits"]
    if output_dir is None:
        output_dir = f"{cfg['training']['output_dir']}-{q_bits}bit-saltq-deploy-eval"

    if model is None:
        assert checkpoint_dir is not None, "export_saltq needs either model or checkpoint_dir"
        model, meta = build_saltq_model(
            saltq_base_dir,
            dtype=getattr(torch, cfg["model"]["dtype"]),
            gradient_checkpointing=False,
            train_scale=bool((cfg["qat"].get("saltq", {}) or {}).get("train_scale", False)),
            continuous_z=bool((cfg["qat"].get("saltq", {}) or {}).get("continuous_z", True)),
        )
        load_saltq_trainable(model, checkpoint_dir)
    else:
        meta = getattr(model, "_saltq_meta", None)
        assert meta is not None, "live model has no _saltq_meta (was prepare_model called?)"

    model = model.to("cpu").eval()
    print("[SALT-Q] Verifying deploy == train equivalence ...")
    verify_saltq_deploy_equivalence(model)

    dtype = _resolve_export_dtype(dequant_dtype)
    print(f"[SALT-Q] Materializing dense deployed weights ({dtype}) ...")
    _, cast_err = materialize_dense_model(model, dtype=dtype)
    print(f"[SALT-Q] dense-storage cast error: max|Δ|={cast_err:.3e} "
          f"(the only lossy step; a packed INT-b export would not have it)")
    gc.collect()

    save_dequantized_model(model, tokenizer, output_dir, cfg, dtype=dtype)
    # The exported model still needs the runtime residual reorder at each segment boundary.
    save_sqat_permute_meta({"layers": {}, "model": meta["perm_meta"]}, output_dir)
    print(f"[SALT-Q] Exported deployed model to {output_dir}")
    return output_dir
