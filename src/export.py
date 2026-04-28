"""
Weight merge & export for QLoRA / QAT / SQAT — asymmetric INT3/INT4 variant.

Export modes:

1) export_dequant=False  (AWQ quantized export)
   dequant_NF4 + LoRA merge -> AWQ-style D-fold -> real PTQ -> save AWQ checkpoint
   - Saves qweight / qzeros / scales / g_idx per linear layer (AutoAWQ format).
   - SQAT salient gain D is folded into the model graph:
       q/k/v    : fold D^-1 into input_layernorm,           D into columns
       o_proj   : fold D^-1 into v_proj output rows,         D into o_proj columns
       gate/up  : fold D^-1 into post_attention_layernorm,   D into columns
       down_proj: D into columns + ScaledActivation(act_fn, D) for D^-1
   - INT3 deployment note: AutoAWQ's GEMM/GEMV kernels are 4-bit only;
     INT3 values fit in [0,7] but are still packed into 4-bit slots.
     Dequant math is identical (shift by zero, multiply by scale), so the
     resulting checkpoint loads on the unmodified AWQ runtime.

2) export_dequant=True  (dense dequantized export)
   dequant_NF4 + LoRA merge -> real PTQ (with D in quantizer) -> dequant -> save dense
   - D is used only inside the quantizer target coordinate system, then the
     dequantized result is mapped back to original weight space before saving.

Asymmetric quant convention (matches training-time fakequant):
    q ∈ [0, q_lvl],  q_lvl = 2^bits − 1
    scale = (w_max − w_min) / q_lvl
    z_int = round(−w_min / scale).clamp(0, q_lvl)
    deq   = (q − z_int) * scale

For SQAT layers, the scale on salient-affected groups is REVERSE-SOLVED from
the salient anchor side (max or min, whichever has larger magnitude), so that
the salient anchor channel reconstructs exactly.  This must mirror the
training-time fakequant byte-for-byte.

Critical invariant:
  PTQ rounding must match training-time fakequant on salient channels.
  _verify_ptq_consistency reports the salient-channel max error explicitly.
"""

import math
import os
import shutil
import tempfile
from collections import OrderedDict, defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from peft import PeftModel

try:
    from awq.modules.act import ScaledActivation as AWQScaledActivation
except Exception:
    AWQScaledActivation = None


# ============================================================================
# ScaledActivation (AWQ or fallback)
# ============================================================================

class FallbackScaledActivation(nn.Module):
    """AWQ-compatible semantics: act(x) / scales."""

    def __init__(self, act_module: nn.Module, scales: torch.Tensor):
        super().__init__()
        self.act = act_module
        self.scales = nn.Parameter(scales.clone().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x) / self.scales


ScaledActivation = AWQScaledActivation or FallbackScaledActivation


# ============================================================================
# Quantization primitives  (asymmetric)
# ============================================================================

def _pad_and_group(W: torch.Tensor, group_size: int):
    """Pad columns to a multiple of group_size and reshape to [out, G, gs]."""
    out_f, in_f = W.shape
    num_groups = math.ceil(in_f / group_size)
    pad = num_groups * group_size - in_f
    W_padded = F.pad(W, (0, pad)) if pad > 0 else W
    return W_padded.view(out_f, num_groups, group_size), num_groups, pad


def _groupwise_quantize_asym(
    W_grouped: torch.Tensor,    # [out, G, gs]
    scales: torch.Tensor,       # [out, G]
    zeros: torch.Tensor,        # [out, G]  integer in [0, q_lvl]
    q_lvl: int,
    in_f: int,
) -> torch.Tensor:
    """Round-clamp-shift quantize grouped weights -> unsigned int in [0, q_lvl]."""
    W_int_grouped = torch.round(
        W_grouped / scales.unsqueeze(2) + zeros.unsqueeze(2).float()
    ).clamp(0, q_lvl).to(torch.int32)
    return W_int_grouped.view(W_grouped.shape[0], -1)[:, :in_f].contiguous()


def real_quantize_asymmetric(
    W: torch.Tensor,
    group_size: int = 128,
    q_bits: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard asymmetric group quantization (no SQAT awareness). CPU only."""
    q_lvl = 2 ** q_bits - 1
    eps = 1e-7
    W = W.float().cpu()
    out_f, in_f = W.shape

    W_grouped, num_groups, _ = _pad_and_group(W, group_size)

    w_max = W_grouped.amax(dim=2)                                      # [out, G]
    w_min = W_grouped.amin(dim=2)
    scales = ((w_max - w_min) / q_lvl).clamp(min=eps)
    zeros = torch.round((-w_min) / scales).clamp(0, q_lvl).to(torch.int32)

    W_int = _groupwise_quantize_asym(W_grouped, scales, zeros, q_lvl, in_f)
    return W_int, scales, zeros


def real_quantize_sqat(
    W_merged: torch.Tensor,
    salient_indices: torch.Tensor,
    base_w_max_group: torch.Tensor,
    base_w_min_group: torch.Tensor,
    salient_group_ids: torch.Tensor,
    salient_gain: Optional[torch.Tensor] = None,
    group_size: int = 128,
    q_bits: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SQAT-aware asymmetric PTQ.  Mirrors training-time fakequant exactly:

      1. amplify salient cols   W_target[:, S] *= D
      2. for groups containing salient channels:
           dynamic w_max/w_min = max/min over (base buffer, amplified salient slice)
           raw_scale = (w_max - w_min) / q_lvl
           z_int     = round(-w_min / raw_scale)
           if salient is max-anchor and |w_max| >= |w_min|:
               scale = w_max / max(q_lvl - z_int, 1)
           elif salient is min-anchor and |w_min| > |w_max|:
               scale = (-w_min) / max(z_int, 1)
           else:
               scale = raw_scale
      3. for non-salient groups: standard asymmetric.

    Returns (W_int, scales, zeros) in the target (amp-)coordinate system.
    """
    q_lvl = 2 ** q_bits - 1
    eps = 1e-7

    # CPU offload (matches real_quantize_asymmetric)
    W_merged = W_merged.float().cpu()
    device = W_merged.device
    salient_indices = salient_indices.to(device)
    salient_group_ids = salient_group_ids.to(device)
    base_w_max_group = base_w_max_group.to(device).float()
    base_w_min_group = base_w_min_group.to(device).float()

    out_f, in_f = W_merged.shape
    num_groups = math.ceil(in_f / group_size)

    # ---- 1. amplify-space target ----
    W_target = W_merged.clone()
    if salient_gain is not None:
        W_target[:, salient_indices] *= salient_gain.to(device).float()

    W_grouped, _, _ = _pad_and_group(W_target, group_size)

    # ---- 2a. dynamic anchor for salient-affected groups ----
    K = salient_indices.shape[0]
    group_indices = salient_group_ids.unsqueeze(0).expand(out_f, K)
    W_salient_amp = W_target[:, salient_indices]          # already amplified

    w_max = base_w_max_group.clone()
    w_max.scatter_reduce_(1, group_indices, W_salient_amp, reduce="amax", include_self=True)
    w_min = base_w_min_group.clone()
    w_min.scatter_reduce_(1, group_indices, W_salient_amp, reduce="amin", include_self=True)

    NEG_INF = torch.finfo(W_target.dtype).min
    POS_INF = torch.finfo(W_target.dtype).max

    w_sal_max = torch.full_like(w_max, NEG_INF)
    w_sal_max.scatter_reduce_(1, group_indices, W_salient_amp, reduce="amax", include_self=True)
    w_sal_min = torch.full_like(w_min, POS_INF)
    w_sal_min.scatter_reduce_(1, group_indices, W_salient_amp, reduce="amin", include_self=True)

    sal_is_max = w_sal_max >= base_w_max_group
    sal_is_min = w_sal_min <= base_w_min_group

    raw_scale_train = ((w_max - w_min) / q_lvl).clamp(min=eps)
    zeros_train = torch.round((-w_min) / raw_scale_train).clamp(0, q_lvl)

    denom_max = (q_lvl - zeros_train).clamp_min(1.0)
    denom_min = zeros_train.clamp_min(1.0)
    scale_from_max = w_max / denom_max
    scale_from_min = (-w_min) / denom_min

    prefer_max_side = w_max.abs() >= w_min.abs()
    use_max = sal_is_max & prefer_max_side
    use_min = sal_is_min & (~prefer_max_side)

    scales_train = raw_scale_train
    scales_train = torch.where(use_max, scale_from_max, scales_train)
    scales_train = torch.where(use_min, scale_from_min, scales_train)
    scales_train = scales_train.clamp(min=eps)

    # ---- 2b. standard asymmetric for non-salient groups (full-group min/max) ----
    w_max_full = W_grouped.amax(dim=2)
    w_min_full = W_grouped.amin(dim=2)
    scales_full = ((w_max_full - w_min_full) / q_lvl).clamp(min=eps)
    zeros_full = torch.round((-w_min_full) / scales_full).clamp(0, q_lvl)

    # ---- 3. select per-group ----
    affected = salient_group_ids.unique()
    group_mask = torch.zeros(num_groups, dtype=torch.bool, device=device)
    group_mask[affected] = True
    gm = group_mask.unsqueeze(0).expand(out_f, num_groups)

    scales = torch.where(gm, scales_train, scales_full)
    zeros = torch.where(gm, zeros_train, zeros_full).to(torch.int32)

    # ---- 4. quantize ----
    W_int = _groupwise_quantize_asym(W_grouped, scales, zeros, q_lvl, in_f)
    return W_int, scales, zeros


def dequantize_asymmetric(
    W_int: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    in_features: int,
) -> torch.Tensor:
    """Dequantize unsigned int weights back to float."""
    out_f = W_int.shape[0]
    num_groups = scales.shape[1]
    pad = num_groups * group_size - in_features
    W_f = W_int.float()
    W_padded = F.pad(W_f, (0, pad)) if pad > 0 else W_f
    W_grouped = W_padded.view(out_f, num_groups, group_size)
    W_deq = (
        (W_grouped - zeros.float().unsqueeze(2)) * scales.float().unsqueeze(2)
    ).view(out_f, -1)
    return W_deq[:, :in_features]


def dequantize_sqat_to_original_space(
    W_int: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    in_features: int,
    group_size: int,
    salient_indices: torch.Tensor,
    salient_gain: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize and undo salient amplification (divide salient cols by D)."""
    W_deq = dequantize_asymmetric(W_int, scales, zeros, group_size, in_features)
    if salient_gain is not None:
        idx = salient_indices.to(W_deq.device)
        W_deq[:, idx] /= salient_gain.to(W_deq.device).float()
    return W_deq


# ============================================================================
# AWQ packing  (4-bit container; INT3 values fit in [0,7])
# ============================================================================

def pack_int4(W_int: torch.Tensor) -> torch.Tensor:
    """
    Pack unsigned 4-bit values (8 per int32) in AWQ column-order.

    Input W_int contains UNSIGNED integers in [0, 15] (INT4) or [0, 7] (INT3).
    For INT3 the high bit is always 0 inside the 4-bit slot — AutoAWQ's GEMM
    kernel still dequantizes correctly because (q − z) * scale is independent
    of the unused high bit.
    """
    out_f, in_f = W_int.shape
    assert in_f % 8 == 0, f"columns={in_f} must be divisible by 8"
    W_uint = W_int.to(torch.int32) & 0xF
    W_packed = W_uint.view(out_f, -1, 8)
    result = torch.zeros(out_f, in_f // 8, dtype=torch.int32, device=W_int.device)
    for i in range(8):
        result |= W_packed[:, :, i] << (i * 4)
    return result


def pack_qzeros(zero_points: torch.Tensor, q_lvl: int = 15) -> torch.Tensor:
    """
    Pack zero points (one per row per group).  Pads with q_lvl // 2 (a
    neutral mid-grid value) — never with the symmetric-INT4 magic number 8,
    which would be out of range for INT3.
    """
    out_f, num_groups = zero_points.shape
    zpad = (8 - num_groups % 8) % 8
    if zpad > 0:
        pad_val = q_lvl // 2
        zero_points = F.pad(zero_points, (0, zpad), value=pad_val)
    return pack_int4(zero_points.to(torch.int32))


# ============================================================================
# SQAT metadata helpers
# ============================================================================

def collect_sqat_metadata(model: nn.Module) -> Dict[str, dict]:
    """Extract salient-channel buffers from all SQAT wrapper layers."""
    from .qat_sqat import SelectiveSalientQATLinear

    metadata = {}
    for name, module in model.named_modules():
        if isinstance(module, SelectiveSalientQATLinear):
            metadata[name] = {
                "salient_indices":   module.salient_indices.cpu().clone(),
                "base_w_max_group":  module.base_w_max_group.cpu().clone(),
                "base_w_min_group":  module.base_w_min_group.cpu().clone(),
                "salient_group_ids": module.salient_group_ids.cpu().clone(),
                "salient_gain":      module.salient_gain.cpu().clone(),
            }
    return metadata


def _unwrap_sqat_for_save(model: nn.Module) -> int:
    """Replace SQAT wrappers with their inner LoRA modules. Returns count."""
    from .qat_sqat import SelectiveSalientQATLinear

    replacements = {
        name: module.original_module
        for name, module in model.named_modules()
        if isinstance(module, SelectiveSalientQATLinear)
    }
    for name, original in replacements.items():
        parts = name.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) == 2 else model
        setattr(parent, parts[-1], original)

    if replacements:
        print(f"[Export] Unwrapped {len(replacements)} SQAT layers for saving")
    return len(replacements)


# ============================================================================
# Internal helpers
# ============================================================================

def _strip_peft_prefix(name: str) -> str:
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _split_block_slot(name: str):
    """Split 'model.layers.3.self_attn.q_proj' -> ('model.layers.3', 'self_attn.q_proj')."""
    parts = name.split(".")
    last_digit_idx = None
    for i, p in enumerate(parts):
        if p.isdigit():
            last_digit_idx = i
    if last_digit_idx is None:
        return None, None
    return ".".join(parts[:last_digit_idx + 1]), ".".join(parts[last_digit_idx + 1:])


def _build_combined_gain(slot_metas: dict, in_features: int, device: torch.device) -> torch.Tensor:
    """Merge per-slot salient gains into a single [in_features] vector (element-wise max)."""
    D = torch.ones(in_features, dtype=torch.float32, device=device)
    for meta in slot_metas.values():
        gain, sidx = meta.get("salient_gain"), meta.get("salient_indices")
        if gain is None or sidx is None:
            continue
        sidx_d = sidx.to(device)
        D[sidx_d] = torch.maximum(D[sidx_d], gain.float().to(device))
    return D


def _get_named_modules(model: nn.Module) -> Dict[str, nn.Module]:
    return dict(model.named_modules())


def _get_linear(modules: dict, name: str) -> Optional[nn.Linear]:
    m = modules.get(name)
    return m if (m is not None and hasattr(m, "weight")) else None


def _is_target_lora(module: nn.Module) -> bool:
    return hasattr(module, "base_layer") and hasattr(module, "lora_A")


def _dequant_base_weight(module: nn.Module) -> torch.Tensor:
    """Extract dequantized base weight [out, in] from a PEFT-wrapped linear."""
    import bitsandbytes as bnb

    base = module.base_layer
    out_f, in_f = base.out_features, base.in_features
    weight = base.weight

    if hasattr(weight, "quant_state") and weight.quant_state is not None:
        W = bnb.functional.dequantize_4bit(weight.data, weight.quant_state).float()
    elif hasattr(base, "dequantize"):
        W = base.dequantize().float()
    else:
        W = weight.data.float()

    if W.shape == (out_f, in_f):
        return W
    if W.shape == (in_f, out_f):
        return W.t().contiguous()
    if W.numel() == out_f * in_f:
        return W.reshape(out_f, in_f)
    raise RuntimeError(f"Cannot reshape dequantized weight {W.shape} to ({out_f}, {in_f})")


# ============================================================================
# Adapter checkpoint key remapping
# ============================================================================

def _remap_adapter_keys_if_needed(adapter_path: str) -> bool:
    """
    If the adapter checkpoint was saved while SQAT wrappers were active,
    LoRA keys contain '.original_module.' and will fail to load onto a clean
    PEFT model. Detect and fix in-place.
    """
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_path, "adapter_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file, save_file
        sd = load_file(safetensors_path)
        if not any(".original_module." in k for k in sd):
            return False
        remapped = {k.replace(".original_module.", "."): v for k, v in sd.items()}
        save_file(remapped, safetensors_path)
    elif os.path.exists(bin_path):
        sd = torch.load(bin_path, map_location="cpu", weights_only=True)
        if not any(".original_module." in k for k in sd):
            return False
        remapped = {k.replace(".original_module.", "."): v for k, v in sd.items()}
        torch.save(remapped, bin_path)
    else:
        return False

    print(f"[Export] Remapped {len(remapped)} adapter keys (stripped .original_module.)")
    return True


# ============================================================================
# AWQ-style salient gain D folding  (UNCHANGED — D-fold is sign-preserving
# linear identity, so it doesn't care whether quant is symmetric or not)
# ============================================================================

def fold_salient_gain_for_awq_export(
    merged_model: nn.Module,
    sqat_meta: Dict[str, dict],
) -> None:
    """
    Fold SQAT salient_gain D into the merged model graph for AWQ export.

    Weight side (D):  W[:, S] *= D   — amplify salient input columns.
    Activation side (D^-1):
      q/k/v    : D^-1 absorbed into input_layernorm.weight
      o_proj   : D^-1 absorbed into v_proj output rows
      gate/up  : D^-1 absorbed into post_attention_layernorm.weight
      down_proj: D^-1 via ScaledActivation wrapping mlp.act_fn
    """
    active = {
        name: meta for name, meta in sqat_meta.items()
        if meta.get("salient_gain") is not None
        and (meta["salient_gain"].float() - 1.0).abs().max().item() > 1e-6
    }
    if not active:
        print("[Export] salient_gain is all-ones; skipping AWQ D-fold.")
        return

    modules = _get_named_modules(merged_model)
    device = next(merged_model.parameters()).device

    QKV_SLOTS = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}
    O_SLOT = "self_attn.o_proj"
    GATE_UP_SLOTS = {"mlp.gate_proj", "mlp.up_proj"}
    DOWN_SLOT = "mlp.down_proj"

    blocks: Dict[str, Dict[str, str]] = defaultdict(dict)
    for name in active:
        bp, slot = _split_block_slot(name)
        if bp is not None:
            blocks[bp][slot] = name

    def _scale_cols(mod, mask, gain):
        if mod is None:
            return
        with torch.no_grad():
            mod.weight.data[:, mask.to(mod.weight.device)] *= gain[mask].to(mod.weight.device)

    def _scale_rows_inv(mod, indices, gain):
        if mod is None:
            return
        with torch.no_grad():
            mod.weight.data[indices.to(mod.weight.device)] /= (
                gain.to(mod.weight.device).unsqueeze(1)
            )

    def _scale_ln_inv(ln, mask, gain):
        if ln is None or not hasattr(ln, "weight"):
            return
        with torch.no_grad():
            ln.weight.data[mask.to(ln.weight.device)] /= gain[mask].to(ln.weight.device)

    for block_prefix, slot_to_name in blocks.items():

        # --- QKV ---
        qkv_active = {s: active[n] for s, n in slot_to_name.items()
                       if s in QKV_SLOTS and n in active}
        if qkv_active:
            rep_mod = _get_linear(modules, slot_to_name[next(iter(qkv_active))])
            if rep_mod is not None:
                D = _build_combined_gain(qkv_active, rep_mod.weight.shape[1], device)
                mask = (D - 1.0).abs() > 1e-6
                if mask.any():
                    _scale_ln_inv(modules.get(f"{block_prefix}.input_layernorm"), mask, D)
                    for slot in qkv_active:
                        _scale_cols(_get_linear(modules, slot_to_name[slot]), mask, D)

        # --- O_proj ---
        if O_SLOT in slot_to_name and slot_to_name[O_SLOT] in active:
            meta = active[slot_to_name[O_SLOT]]
            sidx = meta["salient_indices"].to(device)
            gain = meta["salient_gain"].float().to(device)
            _scale_rows_inv(_get_linear(modules, f"{block_prefix}.self_attn.v_proj"), sidx, gain)
            o_mod = _get_linear(modules, slot_to_name[O_SLOT])
            if o_mod is not None:
                with torch.no_grad():
                    o_mod.weight.data[:, sidx.to(o_mod.weight.device)] *= gain.to(o_mod.weight.device)

        # --- Gate/Up ---
        gu_active = {s: active[n] for s, n in slot_to_name.items()
                      if s in GATE_UP_SLOTS and n in active}
        if gu_active:
            rep_mod = _get_linear(modules, slot_to_name[next(iter(gu_active))])
            if rep_mod is not None:
                D = _build_combined_gain(gu_active, rep_mod.weight.shape[1], device)
                mask = (D - 1.0).abs() > 1e-6
                if mask.any():
                    _scale_ln_inv(modules.get(f"{block_prefix}.post_attention_layernorm"), mask, D)
                    for slot in gu_active:
                        _scale_cols(_get_linear(modules, slot_to_name[slot]), mask, D)

        # --- Down_proj ---
        if DOWN_SLOT in slot_to_name and slot_to_name[DOWN_SLOT] in active:
            meta = active[slot_to_name[DOWN_SLOT]]
            sidx = meta["salient_indices"].to(device)
            gain = meta["salient_gain"].float().to(device)
            down_mod = _get_linear(modules, slot_to_name[DOWN_SLOT])
            if down_mod is not None:
                with torch.no_grad():
                    down_mod.weight.data[:, sidx.to(down_mod.weight.device)] *= gain.to(down_mod.weight.device)

                mlp_mod = modules.get(f"{block_prefix}.mlp")
                if mlp_mod is not None and hasattr(mlp_mod, "act_fn"):
                    act_dim = down_mod.weight.shape[1]
                    full_scales = torch.ones(act_dim, dtype=torch.float32, device=device)
                    full_scales[sidx] = gain
                    if not isinstance(mlp_mod.act_fn, (ScaledActivation, FallbackScaledActivation)):
                        mlp_mod.act_fn = ScaledActivation(mlp_mod.act_fn, full_scales)
                    else:
                        with torch.no_grad():
                            mlp_mod.act_fn.scales.data.copy_(full_scales)

    print(f"[Export] Applied AWQ-style D-fold to {len(active)} SQAT layers.")


# ============================================================================
# PTQ verification
# ============================================================================

def _verify_ptq_consistency(
    weight_targets: Dict[str, torch.Tensor],
    quantized_layers: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    sqat_meta: Dict[str, dict],
    group_size: int,
    max_layers: int = 5,
):
    """
    Spot-check that dequant(quant(W)) ≈ W for a few layers.

    For SQAT layers we additionally check the salient-channel error: salient
    anchors should be near-exact (≤ FP rounding); STE-aligned salient
    channels should be within ±scale/2.  If salient max error blows up, the
    PTQ quantizer is no longer consistent with training-time fakequant.
    """
    checked = 0
    for name, target in weight_targets.items():
        if checked >= max_layers or name not in quantized_layers:
            continue
        W_int, scales, zeros = quantized_layers[name]
        in_f = target.shape[1]

        # If this is a SQAT layer, dequant lives in amp-space; map back.
        meta = sqat_meta.get(name) if sqat_meta else None
        if meta is not None:
            W_deq = dequantize_sqat_to_original_space(
                W_int=W_int, scales=scales, zeros=zeros,
                in_features=in_f, group_size=group_size,
                salient_indices=meta["salient_indices"],
                salient_gain=meta["salient_gain"],
            ).to(target.device)
            target_for_diff = target.float()  # original space
        else:
            W_deq = dequantize_asymmetric(W_int, scales, zeros, group_size, in_f).to(target.device)
            target_for_diff = target.float()

        abs_err = (W_deq - target_for_diff).abs()
        max_err = abs_err.max().item()
        mean_err = abs_err.mean().item()
        status = "OK" if max_err < 0.5 else "WARN"

        salient_msg = ""
        if meta is not None:
            sidx = meta["salient_indices"].to(target.device)
            sal_err = abs_err[:, sidx]
            salient_msg = (
                f"  salient_max={sal_err.max().item():.5f}, "
                f"salient_mean={sal_err.mean().item():.6f}"
            )

        print(f"  [Verify] {name}: max_err={max_err:.4f}, mean_err={mean_err:.6f} [{status}]{salient_msg}")
        checked += 1
    if checked == 0:
        print("  [Verify] No layers checked.")


# ============================================================================
# Save helpers
# ============================================================================

def save_awq_quantized_model(
    model_fp16: nn.Module,
    quantized_layers: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    tokenizer,
    output_dir: str,
    cfg: dict,
):
    """Save AWQ-style int4-container checkpoint (works for INT3 values too)."""
    group_size = cfg["qat"].get("group_size", 128)
    q_bits = cfg["model"]["quant_bits"]
    q_lvl = 2 ** q_bits - 1
    os.makedirs(output_dir, exist_ok=True)

    state_dict = OrderedDict()
    for name, param in model_fp16.named_parameters():
        layer_name = name.rsplit(".weight", 1)[0] if name.endswith(".weight") else None
        if layer_name and layer_name in quantized_layers:
            W_int, scales, zeros = quantized_layers[layer_name]
            in_f = W_int.shape[1]
            pad = (8 - in_f % 8) % 8
            W_pack_src = F.pad(W_int, (0, pad)) if pad > 0 else W_int
            g_idx = torch.arange(W_pack_src.shape[1], dtype=torch.int32) // group_size
            state_dict[f"{layer_name}.qweight"] = pack_int4(W_pack_src).cpu()
            state_dict[f"{layer_name}.qzeros"] = pack_qzeros(zeros, q_lvl=q_lvl).cpu()
            state_dict[f"{layer_name}.scales"] = scales.cpu().half()
            state_dict[f"{layer_name}.g_idx"] = g_idx.cpu()
        else:
            state_dict[name] = param.data.cpu().half()

    try:
        from safetensors.torch import save_file
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    except ImportError:
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    config = AutoConfig.from_pretrained(cfg["model"]["name"])
    config.quantization_config = {
        "quant_method": "awq",
        "zero_point": True,
        "group_size": group_size,
        "bits": q_bits,
        "version": "gemm",
    }
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_dequantized_model(
    model: nn.Module, tokenizer, output_dir: str, cfg: dict,
    dtype: torch.dtype = torch.float16,
):
    """Save as a standard dense model after quantize->dequant."""
    os.makedirs(output_dir, exist_ok=True)
    model = model.to("cpu")
    for p in model.parameters():
        p.data = p.data.to(dtype)

    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        model.config.quantization_config = None

    try:
        model.save_pretrained(output_dir, safe_serialization=True)
    except TypeError:
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# ============================================================================
# LoRA merge into dense model
# ============================================================================

def _merge_lora_into_dense(
    peft_model: PeftModel,
    merged_model: nn.Module,
    target_modules: list,
    lora_scaling: float,
) -> int:
    """Dequant NF4 base + LoRA delta -> copy into dense model. Returns layer count."""
    import bitsandbytes as bnb

    modules_map = _get_named_modules(merged_model)
    count = 0

    for name, module in peft_model.named_modules():
        terminal = name.split(".")[-1] if name else ""
        if terminal not in target_modules or not _is_target_lora(module):
            continue

        W_base = _dequant_base_weight(module)
        out_f, in_f = module.base_layer.out_features, module.base_layer.in_features

        adapter_name = list(module.lora_A.keys())[0]
        A = module.lora_A[adapter_name].weight.data.float()
        B = module.lora_B[adapter_name].weight.data.float()
        W_merged = (W_base + (B @ A) * lora_scaling).half()

        clean_name = _strip_peft_prefix(name)
        target = modules_map.get(clean_name)
        if target is not None and hasattr(target, "weight"):
            target.weight.data.copy_(W_merged.cpu())
            count += 1
        else:
            print(f"[Export] WARNING: target not found for {clean_name}")

    return count


# ============================================================================
# Per-layer quantization loop
# ============================================================================
def _quantize_all_layers(
    merged_model: nn.Module,
    target_modules: list,
    qat_mode: str,
    sqat_meta: Dict[str, dict],
    group_size: int,
    q_bits: int,
    export_dequant: bool,
) -> Tuple[Dict[str, Tuple], Dict[str, torch.Tensor]]:
    """
    Run asymmetric PTQ on all target linear layers.

    For SQAT layers:
      - export_dequant=True:   pass salient_gain=D (quantizer runs in amp-space).
      - export_dequant=False:  D has already been folded into the model graph by
                               fold_salient_gain_for_awq_export(), so gain=None.
    """
    quantized_layers = {}
    quant_targets = {}

    for name, module in tqdm(list(merged_model.named_modules()), desc="Quantizing"):
        if not isinstance(module, nn.Linear):
            continue
        terminal = name.split(".")[-1] if name else ""
        if terminal not in target_modules:
            continue

        W = module.weight.data.float()

        if qat_mode == "sqat" and name in sqat_meta:
            meta = sqat_meta[name]
            gain = meta["salient_gain"] if export_dequant else None

            W_int, scales, zeros = real_quantize_sqat(
                W_merged=W,
                salient_indices=meta["salient_indices"],
                base_w_max_group=meta["base_w_max_group"],
                base_w_min_group=meta["base_w_min_group"],
                salient_group_ids=meta["salient_group_ids"],
                salient_gain=gain,
                group_size=group_size,
                q_bits=q_bits,
            )
        else:
            W_int, scales, zeros = real_quantize_asymmetric(
                W,
                group_size=group_size,
                q_bits=q_bits,
            )

        quantized_layers[name] = (W_int, scales, zeros)
        quant_targets[name] = W.cpu()

    return quantized_layers, quant_targets


# ============================================================================
# Main export pipeline
# ============================================================================

def merge_and_export(
    model: Optional[PeftModel],
    tokenizer,
    cfg: dict,
    checkpoint_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    sqat_metadata: Optional[Dict] = None,
    export_dequant: bool = False,
    dequant_dtype: str = "float16",
):
    """
    Full export pipeline.

    export_dequant=False:
      dequant_NF4 + LoRA merge -> AWQ-style D-fold -> real PTQ -> save AWQ

    export_dequant=True:
      dequant_NF4 + LoRA merge -> real PTQ (with D) -> dequant to original space -> save dense
    """
    qat_mode = cfg["qat"]["mode"]
    q_bits = cfg["model"]["quant_bits"]
    group_size = cfg["qat"].get("group_size", 128)
    base_model_name = cfg["model"]["name"]
    target_modules = cfg["lora"]["target_modules"]
    lora_scaling = cfg["lora"]["alpha"] / cfg["lora"]["rank"]

    suffix = "dequant" if export_dequant else "awq"
    if output_dir is None:
        output_dir = f"{cfg['training']['output_dir']}-{q_bits}bit-{qat_mode}-{suffix}-eval"

    print(f"[Export] Mode: {'dequantized dense' if export_dequant else 'AWQ quantized'}")
    print(f"[Export] QAT mode: {qat_mode}, INT{q_bits} (asymmetric), group_size={group_size}")

    # --- Collect SQAT metadata before unwrap ---
    if qat_mode == "sqat" and sqat_metadata is None and model is not None:
        print("[Export] Collecting SQAT metadata...")
        sqat_metadata = collect_sqat_metadata(model)
        print(f"[Export]   Found {len(sqat_metadata)} SQAT layers")

    # --- Prepare adapter checkpoint ---
    tmp_dir = None
    adapter_path = checkpoint_dir
    if adapter_path is None:
        assert model is not None, "Provide either model or checkpoint_dir"
        tmp_dir = tempfile.mkdtemp(prefix="qlora_export_")
        adapter_path = tmp_dir
        _unwrap_sqat_for_save(model)
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print(f"[Export] Saved adapter to {adapter_path}")
    else:
        _remap_adapter_keys_if_needed(adapter_path)

    # --- Load NF4 base + LoRA adapter ---
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig

    print("[Export] Loading base model in NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["model"].get("quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=cfg["model"].get("double_quant", True),
    )
    base_model_nf4 = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print("[Export] Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(
        base_model_nf4, adapter_path, torch_dtype=torch.float16,
    )

    # --- Merge into dense shell ---
    print("[Export] Loading dense model shell...")
    merged_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    count = _merge_lora_into_dense(peft_model, merged_model, target_modules, lora_scaling)
    print(f"[Export] Merged {count} layers")

    del peft_model, base_model_nf4
    torch.cuda.empty_cache()

    # --- Reindex SQAT metadata to dense-model names ---
    sqat_meta = {}
    if sqat_metadata:
        sqat_meta = {_strip_peft_prefix(k): v for k, v in sqat_metadata.items()}

    # --- AWQ path: fold D into graph; Dequant path: keep on CPU ---
    if not export_dequant:
        merged_model = merged_model.to("cuda")
        if qat_mode == "sqat" and sqat_meta:
            print("[Export] Applying AWQ-style SQAT D-fold...")
            fold_salient_gain_for_awq_export(merged_model, sqat_meta)

    # --- PTQ ---
    print("[Export] Applying asymmetric PTQ...")
    quantized_layers, quant_targets = _quantize_all_layers(
        merged_model, target_modules, qat_mode, sqat_meta,
        group_size, q_bits, export_dequant,
    )
    print(f"[Export] Quantized {len(quantized_layers)} layers")

    print("[Export] Verifying a few layers...")
    _verify_ptq_consistency(quant_targets, quantized_layers,
                            sqat_meta if export_dequant else {},
                            group_size)

    # --- Save ---
    if export_dequant:
        print("[Export] Replacing weights with dequantized dense tensors...")
        modules_map = _get_named_modules(merged_model)
        for name, (W_int, scales, zeros) in quantized_layers.items():
            mod = modules_map.get(name)
            if mod is None or not hasattr(mod, "weight"):
                continue

            in_f = mod.weight.shape[1]

            if qat_mode == "sqat" and name in sqat_meta:
                meta = sqat_meta[name]
                # Quantizer ran in amp-space (we passed gain=D); map back.
                W_deq = dequantize_sqat_to_original_space(
                    W_int=W_int,
                    scales=scales,
                    zeros=zeros,
                    in_features=in_f,
                    group_size=group_size,
                    salient_indices=meta["salient_indices"],
                    salient_gain=meta["salient_gain"],
                )
            else:
                W_deq = dequantize_asymmetric(W_int, scales, zeros, group_size, in_f)

            mod.weight.data.copy_(W_deq.to(mod.weight.dtype))

        dtype = torch.float16 if dequant_dtype.lower() in ("fp16", "float16", "half") else torch.bfloat16
        save_dequantized_model(merged_model, tokenizer, output_dir, cfg, dtype=dtype)
        print(f"[Export] Saved dense quantize->dequant model to {output_dir}")
    else:
        print("[Export] Saving AWQ checkpoint...")
        save_awq_quantized_model(merged_model, quantized_layers, tokenizer, output_dir, cfg)
        print(f"[Export] Saved AWQ checkpoint to {output_dir}")

    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_dir


# ============================================================================
# Adapter-only export
# ============================================================================

def export_adapter_only(
    model: PeftModel,
    tokenizer,
    cfg: dict,
    output_dir: Optional[str] = None,
):
    """Save just the LoRA adapter (unwrap SQAT wrappers first)."""
    if output_dir is None:
        qat_mode = cfg["qat"]["mode"]
        bits = cfg["model"]["quant_bits"]
        output_dir = f"{cfg['training']['output_dir']}-{bits}bit-{qat_mode}-adapter"

    _unwrap_sqat_for_save(model)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Export] Adapter saved to {output_dir}")
    return output_dir


# ============================================================================
# Merged-only export (no quantization — upper bound baseline)
# ============================================================================

def export_merged_only(
    model: Optional[PeftModel],
    tokenizer,
    cfg: dict,
    checkpoint_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    dequant_dtype: str = "float16",
):
    """
    Export NF4-dequant + LoRA merge as dense fp16/bf16 WITHOUT any PTQ.
    Upper-bound baseline: best quality from this LoRA checkpoint.
    """
    qat_mode = cfg.get("qat", {}).get("mode", "none")
    base_model_name = cfg["model"]["name"]
    target_modules = cfg["lora"]["target_modules"]
    lora_scaling = cfg["lora"]["alpha"] / cfg["lora"]["rank"]

    if output_dir is None:
        output_dir = f"{cfg['training']['output_dir']}-merged-noquant"

    print(f"[Export] Mode: merged dense (NO quantization) — upper bound baseline")

    tmp_dir = None
    adapter_path = checkpoint_dir
    if adapter_path is None:
        assert model is not None, "Provide either model or checkpoint_dir"
        tmp_dir = tempfile.mkdtemp(prefix="qlora_export_")
        adapter_path = tmp_dir
        _unwrap_sqat_for_save(model)
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print(f"[Export] Saved adapter to {adapter_path}")
    else:
        _remap_adapter_keys_if_needed(adapter_path)

    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig

    print("[Export] Loading base model in NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["model"].get("quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=cfg["model"].get("double_quant", True),
    )
    base_model_nf4 = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print("[Export] Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(
        base_model_nf4, adapter_path, torch_dtype=torch.float16,
    )

    print("[Export] Loading dense model shell...")
    merged_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    count = _merge_lora_into_dense(peft_model, merged_model, target_modules, lora_scaling)
    print(f"[Export] Merged {count} layers (no quantization applied)")

    del peft_model, base_model_nf4
    torch.cuda.empty_cache()

    dtype = torch.float16 if dequant_dtype.lower() in ("fp16", "float16", "half") else torch.bfloat16
    save_dequantized_model(merged_model, tokenizer, output_dir, cfg, dtype=dtype)
    print(f"[Export] Saved merged dense model (no quant) to {output_dir}")

    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_dir