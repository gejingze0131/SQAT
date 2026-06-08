"""
QAT Interface & Implementations.

Three modes:
  - "none":  standard QLoRA, no quantization-aware training
  - "full":  LR-QAT — fakequant the MERGED weight (frozen NF4 base + trainable LoRA) on every
             forward, i.e. the low-rank term is INSIDE the quantizer (Q(W0 + s·B·A)). All target
             weights are quantized; only LoRA trains.
  - "sqat":  selective salient QAT (only top-k channels by activation 2nd moment)
  - "qalora": group-wise QA-LoRA with affine asymmetric fakequant

Full QAT = LR-QAT (LoRA inside the quantizer)
---------------------------------------------
The forward fake-quantizes  W_curr = dequant(base) + (B @ A) * scaling  — the SAME merged weight
the export pipeline quantizes — so training and deployment share the exact INT4 grid (and the LoRA
contribution is rounded at train time too, not just at export). This matches how Selective-QAT
(qat_permute_sqat) treats its salient slice, making "full vs selective" a clean single-variable
comparison (coverage only). It is implemented as a forward hook per PEFT module that adds
`(fakequant(W_curr) - W_curr) @ x` to the module output (== fakequant(W_curr) @ x); the base is
re-dequantized each forward (no cache, frozen / no grad), only LoRA A/B receive gradients through
the STE. `group_size` defaults to 128 to match export.merge_and_export — the training fakequant
grid MUST equal the PTQ grid at export time. With gradient checkpointing on, the full-weight
materialization is bounded to one decoder layer at a time.
"""

import math
import warnings
from enum import Enum
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QATMode(Enum):
    NONE = "none"
    FULL = "full"
    SQAT = "sqat"
    QALORA = "qalora"
    SQAT_PERMUTE = "sqat_permute"


# ============================================================================
# Core quantization primitives (shared by Full QAT and SQAT)
# ============================================================================

def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator: forward = round, backward = identity."""
    return (torch.round(x) - x).detach() + x


def symmetric_fakequant(w: torch.Tensor, scale: torch.Tensor, q_max: float = 7.0) -> torch.Tensor:
    """Symmetric fake quantization with STE."""
    return round_ste(torch.clamp(w / scale, -q_max, q_max)) * scale


def asymmetric_scale_zero_from_pos_neg(
    pos: torch.Tensor,
    neg: torch.Tensor,
    q_max: int,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build affine quantizer params from positive and negative ranges.

    zero_point is integer-valued.  For two-sided ranges it is clamped inside
    [1, q_max - 1] so both signs remain representable.  The scale is then
    chosen from the active side, which lets at least one range anchor land
    exactly on qmin or qmax after zero-point rounding.
    """
    pos = pos.clamp(min=0.0)
    neg = neg.clamp(min=0.0)

    has_pos = pos > eps
    has_neg = neg > eps
    zp = torch.round(q_max * neg / (pos + neg).clamp(min=eps))
    zp = torch.where(has_pos & has_neg, zp.clamp(1, q_max - 1), zp)
    zp = torch.where(has_pos & ~has_neg, torch.zeros_like(zp), zp)
    zp = torch.where(~has_pos & has_neg, torch.full_like(zp, float(q_max)), zp)
    zp = zp.clamp(0, q_max)

    pos_scale = pos / (q_max - zp).clamp(min=1.0)
    neg_scale = neg / zp.clamp(min=1.0)
    scale = torch.maximum(pos_scale, neg_scale).clamp(min=eps)
    return scale, zp


def asymmetric_scale_zero_from_pos_neg_ultrafast(
    pos: torch.Tensor,
    neg: torch.Tensor,
    q_max: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ultrafast two-sided asymmetric affine quantizer.

    Requires:
        pos > 0
        neg > 0
        1 <= zero_point <= q_max - 1 after rounding/clamping
    """
    q_max = int(q_max)

    zero_point = torch.round(q_max * neg / (pos + neg))
    zero_point = zero_point.clamp_(1, q_max - 1)

    pos_scale = pos / (q_max - zero_point)
    neg_scale = neg / zero_point

    scale = torch.maximum(pos_scale, neg_scale)

    signed_anchor = torch.where(pos_scale >= neg_scale, pos, -neg)

    return scale, zero_point, signed_anchor


def asymmetric_fakequant(
    w: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    q_max: int,
) -> torch.Tensor:
    """Affine fake quantization with STE. Quantized values are in [0, q_max]."""
    q = round_ste(torch.clamp(w / scale + zero_point, 0, q_max))
    return (q - zero_point) * scale


def groupwise_symmetric_fakequant(
    W: torch.Tensor,
    group_size: int,
    q_max: float,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Per-output-row, per-input-group symmetric fakequant.

    W is expected as [out_features, in_features] in fp32. The rounding step
    MUST run in fp32 — doing it in fp16 can silently collapse to a no-op on
    tiny magnitudes and defeat the whole point of QAT.
    """
    out_f, in_f = W.shape
    g = group_size
    num_groups = math.ceil(in_f / g)
    pad = num_groups * g - in_f
    Wp = F.pad(W, (0, pad)) if pad > 0 else W
    Wg = Wp.view(out_f, num_groups, g)
    scale = (Wg.abs().amax(dim=2, keepdim=True) / q_max).clamp(min=eps)
    Wfq = round_ste(torch.clamp(Wg / scale, -q_max, q_max)) * scale
    return Wfq.view(out_f, -1)[:, :in_f]


def groupwise_asymmetric_fakequant(
    W: torch.Tensor,
    group_size: int,
    q_max: int,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Per-output-row, per-input-group affine fakequant."""
    out_f, in_f = W.shape
    g = group_size
    num_groups = math.ceil(in_f / g)
    pad = num_groups * g - in_f
    Wp = F.pad(W, (0, pad)) if pad > 0 else W
    Wg = Wp.view(out_f, num_groups, g)
    pos = Wg.clamp(min=0).amax(dim=2, keepdim=True)
    neg = (-Wg).clamp(min=0).amax(dim=2, keepdim=True)
    scale, zero_point = asymmetric_scale_zero_from_pos_neg(pos, neg, q_max, eps)
    Wfq = asymmetric_fakequant(Wg, scale, zero_point, q_max)
    return Wfq.view(out_f, -1)[:, :in_f]


# ============================================================================
# bnb 4-bit dequant helper
# ============================================================================

def _dequant_bnb_4bit(base_linear: nn.Module) -> torch.Tensor:
    """
    Dequantize a bnb Linear4bit's weight to a dense tensor in compute dtype.
    Always returns shape [out_features, in_features].
    """
    weight = base_linear.weight
    out_f = base_linear.out_features
    in_f = base_linear.in_features

    if hasattr(weight, "quant_state") and weight.quant_state is not None:
        import bitsandbytes.functional as bnbF
        W = bnbF.dequantize_4bit(weight.data, weight.quant_state)
    elif hasattr(weight, "dequantize"):
        # Some bnb variants expose .dequantize() on the param itself.
        W = weight.dequantize()
    else:
        # Do NOT fall through to weight.data.float() — for Params4bit that
        # returns the packed uint8 storage, not real weights.
        raise RuntimeError(
            f"Cannot dequantize base_layer of type {type(base_linear).__name__}; "
            f"expected a bnb Params4bit with quant_state."
        )

    if W.shape == (out_f, in_f):
        return W
    if W.shape == (in_f, out_f):
        return W.t().contiguous()
    if W.numel() == out_f * in_f:
        return W.reshape(out_f, in_f)
    raise RuntimeError(
        f"Dequantized weight shape {tuple(W.shape)} cannot be reshaped to "
        f"({out_f}, {in_f})."
    )


# ============================================================================
# Abstract QAT Handler
# ============================================================================

class QATHandler(ABC):
    """Base class for QAT strategies. Implementations hook into the training loop."""

    @abstractmethod
    def prepare_model(self, model: nn.Module, cfg: dict, **kwargs) -> nn.Module:
        """Called once before training. Patch model if needed."""
        ...

    @abstractmethod
    def on_train_begin(self, model: nn.Module):
        """Called at the start of training."""
        ...

    @abstractmethod
    def on_step_end(self, model: nn.Module, step: int):
        """Called after each optimizer step (for scale refresh, etc.)."""
        ...

    @abstractmethod
    def on_train_end(self, model: nn.Module):
        """Called at the end of training. Cleanup, unwrap, etc."""
        ...


# ============================================================================
# No QAT (standard QLoRA)
# ============================================================================

class NoQAT(QATHandler):
    """Passthrough — standard QLoRA without any QAT."""

    def prepare_model(self, model, cfg, **kwargs):
        return model

    def on_train_begin(self, model):
        pass

    def on_step_end(self, model, step):
        pass

    def on_train_end(self, model):
        pass


# ============================================================================
# Full QAT = LR-QAT — fakequant the MERGED weight (LoRA inside the quantizer)
# ============================================================================

class FullQATLoRAInjector(nn.Module):
    """
    LR-QAT injector for ONE PEFT LoRA module: the low-rank term is INSIDE the quantizer.

    The frozen NF4 base is dequantized on the fly (no grad) and the low-rank term is added BEFORE
    fake-quantization, so the quantizer sees the SAME merged weight the export will quantize. The
    module forward is REPLACED with a SINGLE fakequant GEMM:

        W_curr = dequant(base) + (B @ A) * scaling            # base frozen; B/A trainable
        y      = F.linear(x, fakequant(W_curr), bias)         # == fakequant(W_curr) @ x + bias

    This avoids the original bnb base GEMM + two LoRA GEMMs + a separate delta GEMM (one GEMM per
    layer instead of ~four), and never materializes the `W_fq - W_curr` delta (one fewer fp32
    full-weight tensor). Only LoRA A/B receive gradients, through the STE fakequant, trained against
    the real INT4 grid of the merged weight (train/deploy consistent). Nothing is cached — the base
    is re-dequantized each forward (so the model stays a genuine NF4-base QLoRA, same memory
    footprint as the other methods) — and the full-weight materialization is bounded by gradient
    checkpointing to one decoder layer at a time. remove() restores the original forward.
    """

    def __init__(self, peft_module, group_size, q_bits, symmetric, lora_scaling, where=""):
        super().__init__()
        self.m = peft_module
        self.group_size = int(group_size)
        self.symmetric = bool(symmetric)
        self.q_bits = int(q_bits)
        self.q_max = (2 ** (q_bits - 1) - 1) if symmetric else (2 ** q_bits - 1)
        self.lora_scaling = float(lora_scaling)
        self.adapter = list(peft_module.lora_A.keys())[0]
        p = float(getattr(peft_module.lora_dropout[self.adapter], "p", 0.0) or 0.0)
        if p > 0.0:
            warnings.warn(
                f"[FullQAT] {where}: lora_dropout={p:g} > 0. The single-GEMM LR-QAT forward uses "
                "the dropout-free B@A, so it won't match a dropout-applied LoRA forward; set "
                "lora_dropout=0 for full QAT.",
                RuntimeWarning,
            )
        self._orig_forward = peft_module.forward
        peft_module.forward = self._forward          # replace the module forward (single GEMM)

    def _forward(self, x, *args, **kwargs):
        m = self.m
        with torch.no_grad():
            W_base = _dequant_bnb_4bit(m.base_layer)                 # [out, in], frozen
        A = m.lora_A[self.adapter].weight                            # [r, in]
        B = m.lora_B[self.adapter].weight                            # [out, r]
        # fp32 only where the rounding needs it; the GEMM runs in the activation dtype.
        W_curr = W_base.float() + (B @ A).float() * self.lora_scaling
        if self.symmetric:
            W_fq = groupwise_symmetric_fakequant(W_curr, self.group_size, float(self.q_max))
        else:
            W_fq = groupwise_asymmetric_fakequant(W_curr, self.group_size, int(self.q_max))
        bias = getattr(m.base_layer, "bias", None)
        if bias is not None:
            bias = bias.to(x.dtype)
        return F.linear(x, W_fq.to(x.dtype), bias)

    def remove(self):
        self.m.forward = self._orig_forward


class FullQAT(QATHandler):
    """
    Full QAT in the LR-QAT paradigm: every target linear's MERGED weight (frozen NF4 base +
    trainable LoRA) is fake-quantized on each forward, so the LoRA is trained INSIDE the quantizer
    against the same INT4 grid the export uses — train/deploy consistent. Base stays frozen NF4;
    only LoRA A/B train. (Contrast with Selective-QAT, which fakequants only the salient slice — so
    this is the full-coverage counterpart with the SAME QAT formulation.)

    Implemented as a forward hook per PEFT module (no base_layer surgery), removed at on_train_end.
    """

    def __init__(self):
        self.injectors: list = []

    def prepare_model(self, model, cfg, **kwargs):
        q_bits = cfg["model"]["quant_bits"]
        # Default MUST match export.merge_and_export (also 128). Training fakequant grid must equal
        # the PTQ grid at export time.
        group_size = cfg["qat"].get("group_size", 128)
        symmetric = cfg["qat"].get("symmetric", True)
        lora_scaling = cfg["lora"]["alpha"] / cfg["lora"]["rank"]
        target_modules = set(cfg["lora"]["target_modules"])

        count = 0
        skipped_not_quant = 0
        for name, module in model.named_modules():
            terminal = name.rsplit(".", 1)[-1] if name else ""
            if terminal not in target_modules:
                continue
            if not (hasattr(module, "base_layer") and hasattr(module, "lora_A")):
                continue

            base = module.base_layer
            if not (hasattr(base, "weight")
                    and getattr(base.weight, "quant_state", None) is not None):
                skipped_not_quant += 1
                continue

            self.injectors.append(FullQATLoRAInjector(
                module, group_size=group_size, q_bits=q_bits, symmetric=symmetric,
                lora_scaling=lora_scaling, where=name,
            ))
            count += 1

        print(
            f"[FullQAT] Installed LR-QAT injectors on {count} PEFT modules "
            f"(group_size={group_size}, bits={q_bits}, symmetric={symmetric})."
        )
        if skipped_not_quant:
            print(
                f"[FullQAT] Skipped {skipped_not_quant} target modules whose "
                f"base_layer is not bnb 4-bit (no quant_state)."
            )
        if count == 0:
            print(
                "[FullQAT] WARNING: no modules injected. Check target_modules "
                "and that the model is loaded in 4-bit."
            )
        return model

    def on_train_begin(self, model):
        pass

    def on_step_end(self, model, step):
        pass

    def on_train_end(self, model):
        """Remove the LR-QAT hooks so checkpoint saving and export see a clean bnb Linear4bit."""
        for inj in self.injectors:
            inj.remove()
        if self.injectors:
            print(f"[FullQAT] Removed {len(self.injectors)} LR-QAT injectors.")
        self.injectors = []


# ============================================================================
# Factory
# ============================================================================

def get_qat_handler(cfg: dict) -> QATHandler:
    """Factory: return the appropriate QAT handler based on config."""
    mode_str = cfg["qat"]["mode"]
    mode = QATMode(mode_str)

    if mode == QATMode.NONE:
        return NoQAT()
    elif mode == QATMode.FULL:
        return FullQAT()
    elif mode == QATMode.SQAT:
        from .qat_sqat import SelectiveSalientQAT
        return SelectiveSalientQAT()
    elif mode == QATMode.QALORA:
        from .qalora import QALoRA
        return QALoRA()
    elif mode == QATMode.SQAT_PERMUTE:
        from .qat_permute_sqat import SegmentPermutedSelectiveQAT
        return SegmentPermutedSelectiveQAT()
    else:
        raise ValueError(f"Unknown QAT mode: {mode_str}")
