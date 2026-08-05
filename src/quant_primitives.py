"""
quant_primitives.py — the canonical group-quantization grid (SINGLE SOURCE OF TRUTH).

Every method in this repo that quantizes on the "sqat_permute family" grid — the permuted
Selective-QAT salient slice, the GPTQ non-salient block, SALT-Q's salient segment and its
frozen-code segment, and the export-time PTQ — derives its scale/zero-point from the two
helpers `_sym_scale` / `_asym_qparams` below. Training fakequant, export quantize and the
grid verifier therefore agree BY CONSTRUCTION.

  Asymmetric (affine):  scale = (wmax - wmin) / q_max ;  zp = round(-wmin/scale).clamp(0, q_max)
                        q = round(clamp(w/scale + zp, 0, q_max)) ;  w' = (q - zp) * scale
  Symmetric:            scale = amax / q_max
                        q = round(clamp(w/scale, -q_max, q_max)) ;  w' = q * scale

DO NOT introduce a second formula. `qat_base.asymmetric_scale_zero_from_pos_neg` is a
DIFFERENT (pos/neg) convention used by the older `sqat` mode and the AWQ packer; mixing the
two silently breaks the train<->export grid (this was a real, expensive regression).

The LSQ / LSQ+ family lives in `qat_base` and is reached here through `fixed_scale=`:
  * asymmetric LSQ+ is NUMERICALLY IDENTICAL to the canonical affine grid above at
    `current_minmax` init (same scale/zp formula, same [0, 2^b-1] levels, same (q-z)*s), so a
    freshly-initialized LSQ model sits exactly on the canonical grid;
  * symmetric LSQ uses Qn = -2^(b-1) (one extra negative level) and is NOT interchangeable
    with the canonical symmetric `[-q_max, q_max]` grid.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qat_base import (
    groupwise_lsq_symmetric_fakequant as _lsq_sym_fq,
    groupwise_lsq_asym_fakequant as _lsq_asym_fq,
    lsq_quantize_export_sym as _lsq_export_sym,
    lsq_quantize_export_asym as _lsq_export_asym,
)


# ============================================================================
# Part 8 — Fresh STE group fakequant (input-column groups; per output row, per group)
# ============================================================================

def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward = round, backward = identity."""
    return (torch.round(x) - x).detach() + x


def _asym_q_max(q_bits: int) -> int:
    """Affine (asymmetric) upper level: 2**bits - 1 (15 for INT4, 7 for INT3)."""
    return 2 ** q_bits - 1


def _sym_q_max(q_bits: int) -> int:
    """Symmetric clamp bound: 2**(bits-1) - 1 (7 for INT4, 3 for INT3)."""
    return 2 ** (q_bits - 1) - 1


# ----------------------------------------------------------------------------
# Canonical per-output-row, per-input-group quantization parameters.
#
# THIS IS THE SINGLE SOURCE OF TRUTH for the SQAT-permute grid. Training fakequant,
# the export PTQ, and the grid verifier ALL derive scale/zero_point from these two
# helpers, so the training-time grid and the deployment-time grid are identical by
# construction (the earlier regression was a min/max-vs-pos/neg formula mismatch
# between training and export — never reintroduce a second formula).
#
# Asymmetric (affine) convention:
#   scale = (wmax - wmin) / q_max ;  zp = round(-wmin/scale) clamped to [0, q_max]
#   quantize:   q  = round(clamp(w/scale + zp, 0, q_max))
#   dequantize: w' = (q - zp) * scale
# Symmetric convention:
#   scale = amax / q_max ;  quantize q = round(clamp(w/scale, -q_max, q_max)) ; w' = q*scale
# ----------------------------------------------------------------------------

def _asym_qparams(Wg: torch.Tensor, q_max: int, eps: float = 1e-8):
    """Wg: [..., group_size] (last dim = a quant group). Returns (scale, zp), each [..., 1]."""
    wmin = Wg.amin(dim=-1, keepdim=True)
    wmax = Wg.amax(dim=-1, keepdim=True)
    scale = ((wmax - wmin) / q_max).clamp(min=eps)
    zp    = torch.round(-wmin / scale).clamp(0, q_max)
    return scale, zp


def _sym_scale(Wg: torch.Tensor, q_max: int, eps: float = 1e-8) -> torch.Tensor:
    """Wg: [..., group_size]. Returns scale [..., 1]."""
    return (Wg.abs().amax(dim=-1, keepdim=True) / q_max).clamp(min=eps)


def groupwise_symmetric_fakequant(
    W: torch.Tensor, group_size: int, q_max: int, eps: float = 1e-8,
) -> torch.Tensor:
    """Symmetric per-row, per-group fakequant with STE. W: [out, group_k] → same shape."""
    out_f, gk = W.shape
    assert gk % group_size == 0, f"group_k={gk} must be a multiple of group_size={group_size}"
    Wg    = W.reshape(out_f, gk // group_size, group_size)
    scale = _sym_scale(Wg, q_max, eps)
    q     = round_ste(torch.clamp(Wg / scale, -q_max, q_max))
    return (q * scale).reshape(out_f, gk)


def groupwise_asymmetric_fakequant(
    W: torch.Tensor, group_size: int, q_max: int, eps: float = 1e-8,
) -> torch.Tensor:
    """Affine asymmetric per-row, per-group fakequant with STE. W: [out, group_k] → same shape."""
    out_f, gk = W.shape
    assert gk % group_size == 0, f"group_k={gk} must be a multiple of group_size={group_size}"
    Wg        = W.reshape(out_f, gk // group_size, group_size)
    scale, zp = _asym_qparams(Wg, q_max, eps)
    q  = round_ste(torch.clamp(Wg / scale + zp, 0, q_max))
    return ((q - zp) * scale).reshape(out_f, gk)


def group_fakequant(
    W: torch.Tensor, group_size: int, q_bits: int, symmetric: bool,
    fixed_scale=None,
) -> torch.Tensor:
    """
    Training fakequant dispatch (STE). Returns dequantized W (same shape).

    fixed_scale (LSQ): when given, the per-step min-max scale is replaced by a learned scale[, zp]
    on the LSQ grid (the qat_base single-source-of-truth functions). sym → a scale tensor [out, ng];
    asym → a (scale[out, ng], zp[out, ng]) tuple. The LSQ grid differs from the min-max grid
    (sym Qn=-2^(b-1) vs the min-max -q_max), so train and export MUST both go through these LSQ
    functions — they do (group_quantize/group_dequantize below take the same fixed_scale).
    """
    if fixed_scale is not None:
        if symmetric:
            return _lsq_sym_fq(W, fixed_scale, group_size, q_bits)
        scale, zp = fixed_scale
        return _lsq_asym_fq(W, scale, zp, group_size, q_bits)
    if symmetric:
        return groupwise_symmetric_fakequant(W, group_size, _sym_q_max(q_bits))
    return groupwise_asymmetric_fakequant(W, group_size, _asym_q_max(q_bits))


# ----------------------------------------------------------------------------
# Export-side real quantize / dequantize (NO STE) — share the qparams above.
# group_dequantize(group_quantize(W)) == group_fakequant(W) exactly (verified).
# ----------------------------------------------------------------------------

@torch.no_grad()
def group_quantize(
    W: torch.Tensor, group_size: int, q_bits: int, symmetric: bool, eps: float = 1e-8,
    fixed_scale=None,
):
    """
    Real group quantization for export. Pads in_features to a group multiple internally.

    Returns:
        W_int:  [out, in_features] int levels (float tensor of integers, trimmed to in_features)
        scale:  [out, num_groups]
        zp:     [out, num_groups]   (all zeros for the symmetric branch)

    fixed_scale (LSQ): when given (sym: scale[out, ng]; asym: (scale, zp)), skip the min-max
    amax/_asym_qparams and quantize with the LEARNED scale[, zp] on the LSQ grid via the qat_base
    export quantizers. This is the export half of the train↔export single-source-of-truth: the
    returned (scale, zp) are exactly what group_dequantize/group_fakequant(fixed_scale=...) use.
    """
    out_f, in_f = W.shape
    if fixed_scale is not None:
        if symmetric:
            scale = fixed_scale.float()
            W_int = _lsq_export_sym(W, scale, group_size, q_bits)
            zp = torch.zeros_like(scale)
        else:
            scale, zp_in = fixed_scale
            scale = scale.float()
            W_int, z_int = _lsq_export_asym(W, scale, zp_in.float(), group_size, q_bits)
            zp = z_int
        return W_int.contiguous(), scale, zp
    ng  = math.ceil(in_f / group_size)
    pad = ng * group_size - in_f
    Wp  = F.pad(W, (0, pad)) if pad > 0 else W
    Wg  = Wp.reshape(out_f, ng, group_size)
    if symmetric:
        q_max = _sym_q_max(q_bits)
        scale = _sym_scale(Wg, q_max, eps)
        q     = torch.round(torch.clamp(Wg / scale, -q_max, q_max))
        zp    = torch.zeros_like(scale)
    else:
        q_max     = _asym_q_max(q_bits)
        scale, zp = _asym_qparams(Wg, q_max, eps)
        q = torch.round(torch.clamp(Wg / scale + zp, 0, q_max))
    W_int = q.reshape(out_f, -1)[:, :in_f].contiguous()
    return W_int, scale.squeeze(-1), zp.squeeze(-1)


@torch.no_grad()
def group_dequantize(
    W_int: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor,
    group_size: int, in_features: int, symmetric: bool,
) -> torch.Tensor:
    """Inverse of group_quantize. Returns dequantized W [out, in_features]."""
    out_f = W_int.shape[0]
    ng    = scale.shape[1]
    pad   = ng * group_size - in_features
    qf    = W_int.float()
    qp    = F.pad(qf, (0, pad)) if pad > 0 else qf
    Wg    = qp.reshape(out_f, ng, group_size)
    s     = scale.unsqueeze(-1).float()
    if symmetric:
        Wdq = Wg * s
    else:
        Wdq = (Wg - zp.unsqueeze(-1).float()) * s
    return Wdq.reshape(out_f, -1)[:, :in_features]


def _strip_peft_prefix(name: str) -> str:
    """Strip the PEFT wrapper prefix so training-time names match export dense-model names."""
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


@torch.no_grad()
def collect_lsq_scales_from_model(model: nn.Module) -> Dict[str, dict]:
    """
    Walk the model for proj modules carrying learned LSQ params (lsq_w_scale[, lsq_w_zp]) and
    return {stripped_proj_name: {"scale": [out, n_sal_g], "zp": [out, n_sal_g]}}.

    Keyed by the EXPORT (PEFT-prefix-stripped) name so export's per-module lookup is direct. The
    scales are in the AMPLIFIED space when AWQ is on (training learned them there); export quantizes
    W*S with these scales, then bakes /S into the dense weight — same as the training fakequant.
    """
    out = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lsq_w_scale"):
            entry = {"scale": mod.lsq_w_scale.detach().float().cpu().clone()}
            if hasattr(mod, "lsq_w_zp"):
                entry["zp"] = mod.lsq_w_zp.detach().float().cpu().clone()
            out[_strip_peft_prefix(name)] = entry
    return out


@torch.no_grad()
def verify_permute_quant_consistency(
    W: torch.Tensor, group_k: int, group_size: int, q_bits: int, symmetric: bool,
    awq_s: Optional[torch.Tensor] = None,
    fixed_scale=None,
) -> float:
    """
    Assert the SALIENT slice's training grid == export grid. Returns max|Δ| over [out, group_k]
    between training fakequant and export quantize→dequant on the salient slice. With the shared
    qparams this must be ~0 (fp round-off only). If AWQ-style scaling is used, pass the per-channel
    `awq_s` the slice is quantized in (the amplify+de-amplify cancels, so it stays a self-check of
    the quantizer formulas in the amplified space).

    fixed_scale (LSQ): when given, both the training fakequant and the export quant→dequant use the
    learned scale[, zp] (LSQ grid) — the check then confirms the LSQ train↔export grids match.
    """
    W_s = W[:, :group_k].float()
    if awq_s is not None:
        s = awq_s.to(torch.float32).view(1, -1)
        fq = group_fakequant(W_s * s, group_size, q_bits, symmetric, fixed_scale=fixed_scale) / s
        wi, sc, zp = group_quantize(W_s * s, group_size, q_bits, symmetric, fixed_scale=fixed_scale)
        dq = group_dequantize(wi, sc, zp, group_size, group_k, symmetric) / s
    else:
        fq = group_fakequant(W_s, group_size, q_bits, symmetric, fixed_scale=fixed_scale)
        wi, sc, zp = group_quantize(W_s, group_size, q_bits, symmetric, fixed_scale=fixed_scale)
        dq = group_dequantize(wi, sc, zp, group_size, group_k, symmetric)
    return (fq - dq).abs().max().item()


def lsq_scale_for_module(lsq_scales: Optional[dict], name: str, symmetric: bool):
    """
    Return the learned LSQ fixed_scale for a proj by name, or None.

    sym  → scale tensor [out, n_sal_g]
    asym → (scale [out, n_sal_g], zp [out, n_sal_g])

    `lsq_scales` is keyed by the export (PEFT-prefix-stripped) name. Names are matched by exact
    key, falling back to a suffix match (handles any residual prefix differences).
    """
    if not lsq_scales:
        return None
    entry = lsq_scales.get(name)
    if entry is None:
        # suffix fallback (e.g. caller passes a name with an extra prefix)
        for k, v in lsq_scales.items():
            if name.endswith(k) or k.endswith(name):
                entry = v
                break
    if entry is None:
        return None
    scale = entry["scale"].float()
    if symmetric:
        return scale
    return scale, entry["zp"].float()
