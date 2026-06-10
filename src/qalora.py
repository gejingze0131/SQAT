"""
QA-LoRA support.

QA-LoRA reduces LoRA adaptation freedom to the quantization-group level:
LoRA A consumes one averaged activation per input group instead of every
input channel.  The resulting group-level delta can be expanded back to the
original weight shape and merged before affine asymmetric quantization.
"""

import math
import os
from types import MethodType
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qat_base import (
    QATHandler,
    _dequant_bnb_4bit,
    groupwise_asymmetric_fakequant,
)


def _active_adapter_name(module: nn.Module) -> str:
    active = getattr(module, "active_adapter", None)
    if isinstance(active, str):
        return active
    if isinstance(active, (list, tuple)) and active:
        return active[0]
    keys = list(getattr(module, "lora_A", {}).keys())
    if not keys:
        raise RuntimeError("QA-LoRA target module has no LoRA adapters.")
    return keys[0]


def _group_lengths(in_features: int, group_size: int, device=None) -> torch.Tensor:
    num_groups = math.ceil(in_features / group_size)
    lengths = torch.full(
        (num_groups,), group_size, dtype=torch.float32, device=device,
    )
    remainder = in_features % group_size
    if remainder:
        lengths[-1] = float(remainder)
    return lengths


def pool_qalora_input(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Average-pool the last dimension into non-overlapping QA-LoRA groups."""
    in_features = x.shape[-1]
    num_groups = math.ceil(in_features / group_size)
    pad = num_groups * group_size - in_features
    x_padded = F.pad(x, (0, pad)) if pad else x
    x_grouped = x_padded.view(*x.shape[:-1], num_groups, group_size)
    pooled = x_grouped.sum(dim=-1)
    lengths = _group_lengths(in_features, group_size, device=x.device).to(x.dtype)
    return pooled / lengths


def expand_qalora_delta(
    delta_group: torch.Tensor,
    in_features: int,
    group_size: int,
) -> torch.Tensor:
    """
    Expand a group-level delta [out, G] to [out, in].

    Since the forward uses per-group means, each original input channel in a
    group receives delta_group[:, g] / group_len.
    """
    lengths = _group_lengths(
        in_features, group_size, device=delta_group.device,
    ).to(delta_group.dtype)
    delta_per_channel = delta_group / lengths.unsqueeze(0)
    return torch.repeat_interleave(
        delta_per_channel, repeats=group_size, dim=1,
    )[:, :in_features].contiguous()


def qalora_delta_from_lora(
    module: nn.Module,
    in_features: int,
    group_size: int,
    adapter_name: str = None,
) -> torch.Tensor:
    """Return the full [out, in] QA-LoRA delta for a patched PEFT module."""
    if adapter_name is None:
        adapter_name = _active_adapter_name(module)
    A = module.lora_A[adapter_name].weight.float()
    B = module.lora_B[adapter_name].weight.float()
    scaling = float(module.scaling[adapter_name])
    delta_group = (B @ A) * scaling
    return expand_qalora_delta(delta_group, in_features, group_size)


def is_qalora_module(module: nn.Module) -> bool:
    return bool(getattr(module, "_qalora_patched", False))


def _make_qalora_forward(group_size: int, q_bits: int):
    q_max = 2 ** q_bits - 1

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        adapter_name = _active_adapter_name(self)
        if (
            getattr(self, "disable_adapters", False)
            or adapter_name not in self.lora_A
        ):
            return self.base_layer(x, *args, **kwargs)

        if getattr(self, "merged", False):
            raise RuntimeError("QA-LoRA does not support merged PEFT modules during training.")

        previous_dtype = x.dtype
        with torch.no_grad():
            W_base = _dequant_bnb_4bit(self.base_layer).to(device=x.device).float()

        in_features = self.base_layer.in_features
        delta = qalora_delta_from_lora(
            self, in_features=in_features, group_size=group_size,
            adapter_name=adapter_name,
        ).to(device=x.device)
        W_q = groupwise_asymmetric_fakequant(
            W_base + delta, group_size=group_size, q_max=q_max,
        ).to(previous_dtype)

        bias = getattr(self.base_layer, "bias", None)
        if bias is not None:
            bias = bias.to(device=x.device, dtype=previous_dtype)
        return F.linear(x, W_q, bias).to(previous_dtype)

    return forward


def patch_qalora_model(
    model: nn.Module,
    cfg: dict,
    *,
    patch_forward: bool = True,
    init_lora_A: bool = True,
) -> Tuple[nn.Module, int]:
    """
    Convert target PEFT LoRA modules to QA-LoRA.

    The adapter A projection is resized from [rank, in_features] to
    [rank, ceil(in_features / group_size)].  B is kept unchanged.
    """
    group_size = int(cfg["qat"].get("group_size", 128))
    q_bits = int(cfg["model"]["quant_bits"])
    target_modules = set(cfg["lora"]["target_modules"])
    dropout = float(cfg["lora"].get("dropout", 0.0))
    if dropout != 0.0:
        raise ValueError(
            "QA-LoRA export-faithful training requires lora.dropout=0.0 "
            "because the merged quantized weight cannot represent activation dropout."
        )

    count = 0
    for name, module in model.named_modules():
        terminal = name.rsplit(".", 1)[-1] if name else ""
        if terminal not in target_modules:
            continue
        if not (hasattr(module, "base_layer") and hasattr(module, "lora_A")):
            continue

        in_features = int(module.base_layer.in_features)
        num_groups = math.ceil(in_features / group_size)
        adapter_name = _active_adapter_name(module)
        old_A = module.lora_A[adapter_name]
        rank = int(old_A.out_features)

        if old_A.in_features != num_groups:
            new_A = nn.Linear(num_groups, rank, bias=False)
            new_A = new_A.to(device=old_A.weight.device, dtype=old_A.weight.dtype)
            if init_lora_A:
                nn.init.xavier_uniform_(new_A.weight)
            module.lora_A[adapter_name] = new_A

        module._qalora_group_size = group_size
        module._qalora_q_bits = q_bits
        module._qalora_patched = True
        if patch_forward:
            if not hasattr(module, "_qalora_original_forward"):
                module._qalora_original_forward = module.forward
            module.forward = MethodType(_make_qalora_forward(group_size, q_bits), module)
        count += 1

    return model, count


def load_qalora_adapter(model: nn.Module, adapter_path: str) -> None:
    """Load a QA-LoRA adapter checkpoint after patch_qalora_model resized A."""
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_path, "adapter_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No adapter_model.safetensors/bin under {adapter_path}")

    if any(".original_module." in k for k in state_dict):
        state_dict = {k.replace(".original_module.", "."): v for k, v in state_dict.items()}

    try:
        from peft import set_peft_model_state_dict
        try:
            set_peft_model_state_dict(model, state_dict, adapter_name="default")
        except TypeError:
            set_peft_model_state_dict(model, state_dict)
    except Exception:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        lora_missing = [k for k in missing if "lora_" in k]
        if lora_missing:
            preview = ", ".join(lora_missing[:5])
            raise RuntimeError(f"Failed to load QA-LoRA adapter; missing keys: {preview}")
        if unexpected:
            print(f"[QA-LoRA] Ignored {len(unexpected)} unexpected adapter keys.")


class QALoRA(QATHandler):
    """Group-wise asymmetric QA-LoRA handler."""

    def __init__(self):
        self.patched_count = 0

    def prepare_model(self, model, cfg, **kwargs):
        if cfg["qat"].get("symmetric", False):
            raise ValueError("QA-LoRA only supports affine asymmetric quantization.")
        model, count = patch_qalora_model(model, cfg, patch_forward=True)
        self.patched_count = count
        print(
            f"[QA-LoRA] Patched {count} LoRA layers "
            f"(group_size={cfg['qat'].get('group_size', 128)}, "
            f"bits={cfg['model']['quant_bits']}, asymmetric=True)."
        )
        if count == 0:
            print("[QA-LoRA] WARNING: no layers patched. Check target_modules.")
        return model

    def on_train_begin(self, model):
        pass

    def on_step_end(self, model, step):
        pass

    def on_train_end(self, model, output_dir=None):
        pass
