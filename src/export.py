"""
Weight merge & export for QLoRA / QAT / SQAT.

Export modes:

1) export_dequant=False  (AWQ quantized export)
   dequant_NF4 + LoRA merge -> AWQ-style D-fold -> real PTQ -> save AWQ checkpoint
   - Saves qweight / qzeros / scales / g_idx per linear layer.
   - SQAT salient gain D is folded into the model graph:
       q/k/v    : fold D^-1 into input_layernorm, D into columns
       o_proj   : fold D^-1 into v_proj output rows, D into o_proj columns
       gate/up  : fold D^-1 into post_attention_layernorm, D into columns
       down_proj: D into columns + ScaledActivation(act_fn, D) for D^-1

2) export_dequant=True  (dense dequantized export)
   dequant_NF4 + LoRA merge -> real PTQ (with D in quantizer) -> dequant -> save dense
   - D is used only inside the quantizer target coordinate system, then the
     dequantized result is mapped back to original weight space before saving.

Critical invariant:
  PTQ rounding must match training-time fakequant:
    round(clamp(w / scale, -q_max, q_max))
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
AWQ_ZERO_POINT = 8


# ============================================================================
# Quantization primitives
# ============================================================================

def _pad_and_group(W: torch.Tensor, group_size: int):
    """Pad columns to a multiple of group_size and reshape to [out, G, gs]."""
    out_f, in_f = W.shape
    num_groups = math.ceil(in_f / group_size)
    pad = num_groups * group_size - in_f
    W_padded = F.pad(W, (0, pad)) if pad > 0 else W
    return W_padded.view(out_f, num_groups, group_size), num_groups, pad


def _groupwise_quantize(W_grouped: torch.Tensor, scales: torch.Tensor,
                        q_max: int, in_f: int):
    """Round-clamp quantize grouped weights, return [out, in_f] int8."""
    W_int_grouped = torch.round(
        torch.clamp(W_grouped / scales.unsqueeze(2), -q_max, q_max)
    ).to(torch.int8)
    return W_int_grouped.view(W_grouped.shape[0], -1)[:, :in_f].contiguous()


def real_quantize_symmetric(
    W: torch.Tensor,
    group_size: int = 128,
    q_bits: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard symmetric group quantization. All work done on CPU."""
    q_max = 2 ** (q_bits - 1) - 1
    W = W.float().cpu()
    out_f, in_f = W.shape

    W_grouped, num_groups, _ = _pad_and_group(W, group_size)
    scales = (W_grouped.abs().amax(dim=2) / q_max).clamp(min=1e-7)
    W_int = _groupwise_quantize(W_grouped, scales, q_max, in_f)

    zeros = torch.full_like(scales, float(AWQ_ZERO_POINT))
    return W_int, scales, zeros


def real_quantize_sqat(
    W_merged: torch.Tensor,
    salient_indices: torch.Tensor,
    base_max_group: torch.Tensor,
    salient_group_ids: torch.Tensor,
    salient_gain: Optional[torch.Tensor] = None,
    group_size: int = 128,
    q_bits: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SQAT-aware PTQ with dynamic-anchor scales on salient groups.

    If salient_gain D is provided, salient columns are amplified before
    quantization:  W_target[:, S] = W_merged[:, S] * D.

    Returns (W_int, scales, zeros) in the target coordinate system.
    """
    q_max = 2 ** (q_bits - 1) - 1

    # [FIX] Offload to CPU to avoid GPU OOM — matches real_quantize_symmetric.
    W_merged = W_merged.float().cpu()
    device = W_merged.device
    salient_indices = salient_indices.to(device)
    salient_group_ids = salient_group_ids.to(device)
    base_max_group = base_max_group.to(device).float()

    out_f, in_f = W_merged.shape
    num_groups = math.ceil(in_f / group_size)

    # Build quantizer target: amplify salient columns if D is provided
    W_target = W_merged.clone()
    if salient_gain is not None:
        W_target[:, salient_indices] *= salient_gain.to(device).float()

    W_grouped, _, _ = _pad_and_group(W_target, group_size)

    # Full-group scales (fallback for non-salient groups)
    scales_full = (W_grouped.abs().amax(dim=2) / q_max).clamp(min=1e-7)

    # Dynamic-anchor scales for salient-affected groups:
    # scale = max(salient_amp_max, base_max) / q_max
    K = salient_indices.shape[0]
    group_indices = salient_group_ids.unsqueeze(0).expand(out_f, K)
    abs_W_salient = W_target[:, salient_indices].abs()

    max_salient = torch.zeros(out_f, num_groups, device=device, dtype=W_target.dtype)
    max_salient.scatter_reduce_(
        dim=1, index=group_indices, src=abs_W_salient,
        reduce="amax", include_self=True,
    )
    scales_train = (torch.maximum(max_salient, base_max_group) / q_max).clamp(min=1e-7)

    # Use training-consistent scales only for groups containing salient channels
    affected = salient_group_ids.unique()
    group_mask = torch.zeros(num_groups, dtype=torch.bool, device=device)
    group_mask[affected] = True
    scales = torch.where(group_mask.unsqueeze(0).expand_as(scales_full),
                         scales_train, scales_full)

    W_int = _groupwise_quantize(W_grouped, scales, q_max, in_f)
    zeros = torch.full_like(scales, float(AWQ_ZERO_POINT))
    return W_int, scales, zeros


def dequantize_symmetric(
    W_int: torch.Tensor, scales: torch.Tensor,
    group_size: int, in_features: int,
) -> torch.Tensor:
    """Dequantize int weights back to float using group scales."""
    out_f = W_int.shape[0]
    num_groups = scales.shape[1]
    pad = num_groups * group_size - in_features
    W_f = W_int.float()
    W_padded = F.pad(W_f, (0, pad)) if pad > 0 else W_f
    W_deq = (
        W_padded.view(out_f, num_groups, group_size) * scales.float().unsqueeze(2)
    ).view(out_f, -1)
    return W_deq[:, :in_features]


def dequantize_sqat_to_original_space(
    W_int: torch.Tensor, scales: torch.Tensor,
    in_features: int, group_size: int,
    salient_indices: torch.Tensor,
    salient_gain: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize and undo salient amplification (divide salient cols by D)."""
    W_deq = dequantize_symmetric(W_int, scales, group_size, in_features)
    if salient_gain is not None:
        idx = salient_indices.to(W_deq.device)
        W_deq[:, idx] /= salient_gain.to(W_deq.device).float()
    return W_deq


# ============================================================================
# INT4 packing for AWQ
# ============================================================================

def pack_int4(W_int: torch.Tensor) -> torch.Tensor:
    """Pack int4 values (8 per int32) in AWQ column-order."""
    out_f, in_f = W_int.shape
    assert in_f % 8 == 0, f"columns={in_f} must be divisible by 8"
    W_uint = (W_int.to(torch.int32) + AWQ_ZERO_POINT) & 0xF
    W_packed = W_uint.view(out_f, -1, 8)
    result = torch.zeros(out_f, in_f // 8, dtype=torch.int32, device=W_int.device)
    for i in range(8):
        result |= W_packed[:, :, i] << (i * 4)
    return result


def pack_qzeros(zero_points: torch.Tensor) -> torch.Tensor:
    out_f, num_groups = zero_points.shape
    zpad = (8 - num_groups % 8) % 8
    if zpad > 0:
        zero_points = F.pad(zero_points, (0, zpad), value=AWQ_ZERO_POINT)
    return pack_int4(zero_points.to(torch.int8))


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
                "base_max_group":    module.base_max_group.cpu().clone(),
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
    [FIX] If the adapter checkpoint was saved while SQAT wrappers were active,
    LoRA keys contain '.original_module.' and will fail to load onto a clean
    PEFT model. Detect and fix in-place.

    Returns True if remapping was performed.
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
# AWQ-style salient gain D folding
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
    # Filter to layers with non-trivial gain
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

    # Group active layers by transformer block
    blocks: Dict[str, Dict[str, str]] = defaultdict(dict)
    for name in active:
        bp, slot = _split_block_slot(name)
        if bp is not None:
            blocks[bp][slot] = name

    def _scale_cols(mod, mask, gain):
        """W[:, mask] *= D[mask]  (amplify input columns)."""
        if mod is None:
            return
        with torch.no_grad():
            mod.weight.data[:, mask.to(mod.weight.device)] *= gain[mask].to(mod.weight.device)

    def _scale_rows_inv(mod, indices, gain):
        """W[indices, :] /= D  (fold D^-1 into output rows)."""
        if mod is None:
            return
        with torch.no_grad():
            mod.weight.data[indices.to(mod.weight.device)] /= (
                gain.to(mod.weight.device).unsqueeze(1)
            )

    def _scale_ln_inv(ln, mask, gain):
        """ln.weight[mask] /= D[mask]  (fold D^-1 into LayerNorm)."""
        if ln is None or not hasattr(ln, "weight"):
            return
        with torch.no_grad():
            ln.weight.data[mask.to(ln.weight.device)] /= gain[mask].to(ln.weight.device)

    for block_prefix, slot_to_name in blocks.items():

        # --- QKV: fold D^-1 into input_layernorm, D into each proj's columns ---
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

        # --- O_proj: fold D^-1 into v_proj rows, D into o_proj columns ---
        if O_SLOT in slot_to_name and slot_to_name[O_SLOT] in active:
            meta = active[slot_to_name[O_SLOT]]
            sidx = meta["salient_indices"].to(device)
            gain = meta["salient_gain"].float().to(device)
            _scale_rows_inv(_get_linear(modules, f"{block_prefix}.self_attn.v_proj"), sidx, gain)
            o_mod = _get_linear(modules, slot_to_name[O_SLOT])
            if o_mod is not None:
                with torch.no_grad():
                    o_mod.weight.data[:, sidx.to(o_mod.weight.device)] *= gain.to(o_mod.weight.device)

        # --- Gate/Up: fold D^-1 into post_attention_layernorm, D into columns ---
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

        # --- Down_proj: D into columns, ScaledActivation(act_fn, D) for D^-1 ---
        if DOWN_SLOT in slot_to_name and slot_to_name[DOWN_SLOT] in active:
            meta = active[slot_to_name[DOWN_SLOT]]
            sidx = meta["salient_indices"].to(device)
            gain = meta["salient_gain"].float().to(device)
            down_mod = _get_linear(modules, slot_to_name[DOWN_SLOT])
            if down_mod is not None:
                with torch.no_grad():
                    down_mod.weight.data[:, sidx.to(down_mod.weight.device)] *= gain.to(down_mod.weight.device)

                # Insert ScaledActivation: act(x) / D  produces D^-1 on activations
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
    group_size: int,
    max_layers: int = 5,
):
    """Spot-check that dequant(quant(W)) ≈ W for a few layers."""
    checked = 0
    for name, target in weight_targets.items():
        if checked >= max_layers or name not in quantized_layers:
            continue
        W_int, scales, _ = quantized_layers[name]
        in_f = target.shape[1]
        W_deq = dequantize_symmetric(W_int, scales, group_size, in_f).to(target.device)
        abs_err = (W_deq - target.float()).abs()
        max_err = abs_err.max().item()
        mean_err = abs_err.mean().item()
        status = "OK" if max_err < 0.5 else "WARN"
        print(f"  [Verify] {name}: max_err={max_err:.4f}, mean_err={mean_err:.6f} [{status}]")
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
    """Save AWQ-style int4 checkpoint with qweight/qzeros/scales/g_idx."""
    group_size = cfg["qat"].get("group_size", 128)
    q_bits = cfg["model"]["quant_bits"]
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
            state_dict[f"{layer_name}.qzeros"] = pack_qzeros(zeros).cpu()
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
    Run PTQ on all target linear layers.

    For SQAT layers:
      - export_dequant=True:
          use the SQAT-aware quantizer in the SAME amplify-space used by training
          (i.e. pass salient_gain=D), then dequantize back to original space later.
      - export_dequant=False:
          D has already been folded into the model graph by fold_salient_gain_for_awq_export(),
          so use gain=None here.

    Returns:
        quantized_layers: {name: (W_int, scales, zeros)}  — all on CPU
        quant_targets:    {name: W_original}               — original-space W for verification
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

            # Dequant export must quantize in the same amplify-space used during SQAT training.
            # AWQ export has already folded D into the graph, so do NOT pass gain again.
            gain = meta["salient_gain"] if export_dequant else None

            W_int, scales, zeros = real_quantize_sqat(
                W_merged=W,
                salient_indices=meta["salient_indices"],
                base_max_group=meta["base_max_group"],
                salient_group_ids=meta["salient_group_ids"],
                salient_gain=gain,
                group_size=group_size,
                q_bits=q_bits,
            )
        else:
            W_int, scales, zeros = real_quantize_symmetric(
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
      dequant_NF4 + LoRA merge -> real PTQ (with D) -> dequant -> save dense
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
    print(f"[Export] QAT mode: {qat_mode}, INT{q_bits}, group_size={group_size}")

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
        # [FIX] Remap keys if checkpoint was saved with SQAT wrappers active
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

    # --- AWQ path: fold D into graph, then move to GPU for packing ---
    # --- Dequant path: keep on CPU, D is handled inside quantizer ---
    if not export_dequant:
        merged_model = merged_model.to("cuda")
        if qat_mode == "sqat" and sqat_meta:
            print("[Export] Applying AWQ-style SQAT D-fold...")
            fold_salient_gain_for_awq_export(merged_model, sqat_meta)
    # [FIX] export_dequant: do NOT move to CUDA — quantization runs on CPU
    #       (real_quantize_sqat now offloads to CPU internally)

    # --- PTQ ---
    print("[Export] Applying PTQ...")
    quantized_layers, quant_targets = _quantize_all_layers(
        merged_model, target_modules, qat_mode, sqat_meta,
        group_size, q_bits, export_dequant,
    )
    print(f"[Export] Quantized {len(quantized_layers)} layers")

    print("[Export] Verifying a few layers...")
    _verify_ptq_consistency(quant_targets, quantized_layers, group_size)

    # --- Save ---
    if export_dequant:
        print("[Export] Replacing weights with dequantized dense tensors...")
        modules_map = _get_named_modules(merged_model)
        for name, (W_int, scales, _) in quantized_layers.items():
            mod = modules_map.get(name)
            if mod is None or not hasattr(mod, "weight"):
                continue

            in_f = mod.weight.shape[1]

            if qat_mode == "sqat" and name in sqat_meta:
                meta = sqat_meta[name]
                W_deq = dequantize_sqat_to_original_space(
                    W_int=W_int,
                    scales=scales,
                    in_features=in_f,
                    group_size=group_size,
                    salient_indices=meta["salient_indices"],
                    salient_gain=meta["salient_gain"],
                )
            else:
                W_deq = dequantize_symmetric(
                    W_int=W_int,
                    scales=scales,
                    group_size=group_size,
                    in_features=in_f,
                )

            mod.weight.data.copy_(W_deq.to(mod.weight.dtype))
            
        print("[Export] Replacing weights with dequantized dense tensors...")
        modules_map = _get_named_modules(merged_model)
        for name, (W_int, scales, _) in quantized_layers.items():
            mod = modules_map.get(name)
            if mod is None or not hasattr(mod, "weight"):
                continue
            in_f = mod.weight.shape[1]
            # Standard dequant for all layers — D was never applied in the
            # quantizer for export_dequant mode, so no D^{-1} mapping needed.
            W_deq = dequantize_symmetric(W_int, scales, group_size, in_f)
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

    This gives the upper-bound baseline: the best quality achievable from
    this LoRA checkpoint, limited only by NF4 dequantization noise in the
    base weights.  No INT4 quantization is applied.

    Usage:
        export_merged_only(model, tokenizer, cfg)
        # or from a saved checkpoint:
        export_merged_only(None, tokenizer, cfg, checkpoint_dir="path/to/adapter")
    """
    qat_mode = cfg.get("qat", {}).get("mode", "none")
    base_model_name = cfg["model"]["name"]
    target_modules = cfg["lora"]["target_modules"]
    lora_scaling = cfg["lora"]["alpha"] / cfg["lora"]["rank"]

    if output_dir is None:
        output_dir = f"{cfg['training']['output_dir']}-merged-noquant"

    print(f"[Export] Mode: merged dense (NO quantization) — upper bound baseline")

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

    # --- Merge into dense shell (NO quantization) ---
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

    # --- Save directly ---
    dtype = torch.float16 if dequant_dtype.lower() in ("fp16", "float16", "half") else torch.bfloat16
    save_dequantized_model(merged_model, tokenizer, output_dir, cfg, dtype=dtype)
    print(f"[Export] Saved merged dense model (no quant) to {output_dir}")

    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_dir
