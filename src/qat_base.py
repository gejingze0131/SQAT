"""
QAT Interface & Implementations.

Three modes:
  - "none":  standard QLoRA, no quantization-aware training
  - "full":  fakequant on ALL quantized weights each forward pass
  - "sqat":  selective salient QAT (only top-k channels by activation 2nd moment)
  - "sqat_bilateral": fixed-grid SQAT on Fisher-selected input/output channels
  - "qalora": group-wise QA-LoRA with affine asymmetric fakequant

Fix notes vs previous version
-----------------------------
The previous FullQAT installed a forward pre-hook that computed a fakequant
weight and wrote it into a cache attribute of a dangling Python object. The
PEFT module's forward kept running against the original NF4-dequantized
weight, so Full QAT training was silently equivalent to NoQAT. The dequant
fallback also returned packed uint8 storage instead of real weights for bnb
Params4bit.

This version:
  * Replaces each PEFT LoRA module's `base_layer` (a bnb Linear4bit) with
    a real nn.Module wrapper whose forward dequantizes NF4, applies
    group-wise symmetric fake-quantization (STE, fp32 rounding), and runs
    the matmul. LoRA A/B sit on the outer PEFT module unchanged and
    learn to compensate for the INT4 grid.
  * Uses `bitsandbytes.functional.dequantize_4bit(w.data, w.quant_state)`
    to dequantize — the only correct path for bnb 4-bit params.
  * Restores the original base_layer in `on_train_end` so checkpoint
    saving and the export pipeline see a normal bnb Linear4bit.
  * Defaults `group_size` to 128 to match `export.merge_and_export`;
    the training fakequant grid MUST equal the PTQ grid at export time.
"""

import math
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QATMode(Enum):
    NONE = "none"
    FULL = "full"
    SQAT = "sqat"
    SQAT_BILATERAL = "sqat_bilateral"
    QALORA = "qalora"


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
# Full QAT — wrapper module that replaces PEFT's base_layer
# ============================================================================

class FullQATBaseLayer(nn.Module):
    """
    Drop-in replacement for a PEFT LoRA module's `base_layer` (bnb Linear4bit).

    On every forward:
      1) dequantize the NF4 base weight to compute dtype (no grad; base is frozen),
      2) apply group-wise symmetric fake-quantization (STE, fp32 rounding),
      3) cast back to activation dtype,
      4) run F.linear(x, W_fq, bias).

    LoRA A / B live on the OUTER PEFT module and are added as a delta on top
    of base_layer(x); their gradients are produced normally and learn to
    compensate for the INT4 quantization grid that now appears in the
    forward graph.
    """

    def __init__(
        self,
        orig_linear4bit: nn.Module,
        group_size: int = 128,
        q_bits: int = 4,
    ):
        super().__init__()
        # Keep the original bnb layer as a submodule so we retain quant_state,
        # bias, shapes, and so restoring at on_train_end is a single attr swap.
        self.orig = orig_linear4bit
        self.group_size = int(group_size)
        self.q_bits = int(q_bits)
        self.q_max = float(2 ** (q_bits - 1) - 1)
        self.in_features = orig_linear4bit.in_features
        self.out_features = orig_linear4bit.out_features

    # PEFT sometimes reads these directly off base_layer for introspection.
    @property
    def weight(self):
        return self.orig.weight

    @property
    def bias(self):
        return getattr(self.orig, "bias", None)

    def _fakequant_weight(self, ref_dtype: torch.dtype) -> torch.Tensor:
        # Dequant under no_grad — base weight is frozen, no grad target upstream.
        with torch.no_grad():
            W = _dequant_bnb_4bit(self.orig)  # [out, in], compute dtype
        # Rounding MUST be fp32.
        W_fq_f32 = groupwise_symmetric_fakequant(
            W.float(), self.group_size, self.q_max
        )
        return W_fq_f32.to(ref_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_fq = self._fakequant_weight(x.dtype)
        return F.linear(x, W_fq, self.bias)


class FullQATBaseLayerAsymmetric(FullQATBaseLayer):
    """Affine asymmetric variant of FullQATBaseLayer."""

    def __init__(
        self,
        orig_linear4bit: nn.Module,
        group_size: int = 128,
        q_bits: int = 4,
    ):
        super().__init__(orig_linear4bit, group_size=group_size, q_bits=q_bits)
        self.q_max = 2 ** q_bits - 1

    def _fakequant_weight(self, ref_dtype: torch.dtype) -> torch.Tensor:
        with torch.no_grad():
            W = _dequant_bnb_4bit(self.orig)
        W_fq_f32 = groupwise_asymmetric_fakequant(
            W.float(), self.group_size, int(self.q_max)
        )
        return W_fq_f32.to(ref_dtype)


class FullQAT(QATHandler):
    """
    Full QAT: every target linear's frozen base weight is fakequanted on
    each forward, so the LoRA adapter is trained against the actual INT4
    quantization grid rather than the fp16 dequantization of NF4.

    Implementation: replace `module.base_layer` on each PEFT LoRA module
    with a FullQATBaseLayer wrapper. Restore at on_train_end so
    checkpoint saving and the export pipeline see a normal bnb Linear4bit.
    """

    def __init__(self):
        # name -> (peft_module, original_base_layer)
        self._originals: Dict[str, Tuple[nn.Module, nn.Module]] = {}

    def prepare_model(self, model, cfg, **kwargs):
        q_bits = cfg["model"]["quant_bits"]
        # Default MUST match export.merge_and_export (also 128). If you change
        # one, change the other — training fakequant grid must equal the PTQ
        # grid at export time.
        group_size = cfg["qat"].get("group_size", 128)
        symmetric = cfg["qat"].get("symmetric", True)
        wrapper_cls = FullQATBaseLayer if symmetric else FullQATBaseLayerAsymmetric
        target_modules = set(cfg["lora"]["target_modules"])

        count = 0
        skipped_not_quant = 0
        for name, module in model.named_modules():
            terminal = name.rsplit(".", 1)[-1] if name else ""
            if terminal not in target_modules:
                continue
            if not (hasattr(module, "base_layer") and hasattr(module, "lora_A")):
                continue

            orig_base = module.base_layer
            # Only wrap if the base is actually bnb 4-bit quantized.
            if not (hasattr(orig_base, "weight")
                    and hasattr(orig_base.weight, "quant_state")
                    and orig_base.weight.quant_state is not None):
                skipped_not_quant += 1
                continue

            wrapper = wrapper_cls(
                orig_base, group_size=group_size, q_bits=q_bits,
            )
            # Assigning a Module to an attribute of an nn.Module (the PEFT
            # layer) correctly re-registers the submodule.
            module.base_layer = wrapper
            self._originals[name] = (module, orig_base)
            count += 1

        print(
            f"[FullQAT] Wrapped {count} PEFT base_layers with fakequant "
            f"(group_size={group_size}, bits={q_bits}, symmetric={symmetric})."
        )
        if skipped_not_quant:
            print(
                f"[FullQAT] Skipped {skipped_not_quant} target modules whose "
                f"base_layer is not bnb 4-bit (no quant_state)."
            )
        if count == 0:
            print(
                "[FullQAT] WARNING: no layers wrapped. Check target_modules "
                "and that the model is loaded in 4-bit."
            )
        return model

    def on_train_begin(self, model):
        pass

    def on_step_end(self, model, step):
        pass

    def on_train_end(self, model):
        """
        Restore the original bnb base_layer on every wrapped PEFT module.

        Essential: if we leave FullQATBaseLayer in place, the wrapper's
        submodule tree (`.orig.weight`, etc.) leaks into any state_dict and
        export._dequant_base_weight will no longer find `quant_state` at the
        expected attribute path.
        """
        for name, (peft_mod, orig_base) in self._originals.items():
            peft_mod.base_layer = orig_base
        if self._originals:
            print(f"[FullQAT] Restored {len(self._originals)} base_layers.")
        self._originals.clear()


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
    elif mode == QATMode.SQAT_BILATERAL:
        from .qat_sqat_bilateral import BilateralSalientQAT
        return BilateralSalientQAT()
    elif mode == QATMode.QALORA:
        from .qalora import QALoRA
        return QALoRA()
    else:
        raise ValueError(f"Unknown QAT mode: {mode_str}")
