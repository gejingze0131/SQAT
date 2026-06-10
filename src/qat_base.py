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
from typing import Optional, Tuple

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
# LSQ / LSQ+ learnable-scale quantization (LR-QAT style)
# ============================================================================
#
# These are a SELF-CONTAINED single source of truth: the scale (and, for asym,
# the zero-point) are learnable nn.Parameters of the model tree, trained jointly
# with the LoRA adapter. The training fakequant and the export quantizer below
# use the SAME Qn/Qp grid so the deployed INT weights are bit-identical to what
# training saw.
#
# Grid convention (DO NOT mix with the min-max symmetric `[-q_max, q_max]`):
#   symmetric  LSQ : Qn = -2^(b-1),   Qp = 2^(b-1) - 1   (one extra negative level)
#   asymmetric LSQ+: Qn = 0,          Qp = 2^b - 1
#
# Initialization is `current_minmax` so that with enable_lsq the initial grid
# equals the original min-max grid (no scale jump at step 0).

def grad_scale(x: torch.Tensor, g: float) -> torch.Tensor:
    """forward = x ; backward: grad *= g (LSQ scale-gradient damping)."""
    return (x - x * g).detach() + x * g


def _group_reshape(W: torch.Tensor, group_size: int) -> torch.Tensor:
    """Pad columns to a multiple of group_size and reshape to [out, ng, gs]."""
    out_f, in_f = W.shape
    num_groups = math.ceil(in_f / group_size)
    pad = num_groups * group_size - in_f
    Wp = F.pad(W, (0, pad)) if pad > 0 else W
    return Wp.view(out_f, num_groups, group_size)


def _ungroup(Wg: torch.Tensor, in_f: int) -> torch.Tensor:
    """Inverse of _group_reshape: [out, ng, gs] -> [out, in_f] (drop padding)."""
    out_f = Wg.shape[0]
    return Wg.reshape(out_f, -1)[:, :in_f]


def _lsq_qn_qp(q_bits: int, symmetric: bool) -> Tuple[int, int]:
    if symmetric:
        return -(2 ** (q_bits - 1)), 2 ** (q_bits - 1) - 1
    return 0, 2 ** q_bits - 1


def groupwise_lsq_symmetric_fakequant(
    W: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    q_bits: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Symmetric LSQ fakequant (only scale is learnable).

    W:     [out, in] fp32
    scale: [out, ng] (>0, learnable nn.Parameter)
    """
    out_f, in_f = W.shape
    Qn, Qp = _lsq_qn_qp(q_bits, symmetric=True)
    Wg = _group_reshape(W, group_size)                              # [out, ng, gs]
    gfactor = 1.0 / math.sqrt(group_size * max(Qp, 1))
    s = grad_scale(scale.clamp(min=eps)[..., None], gfactor)        # [out, ng, 1]
    Wq = round_ste((Wg / s).clamp(Qn, Qp)) * s
    return _ungroup(Wq, in_f)


def groupwise_lsq_asym_fakequant(
    W: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
    group_size: int,
    q_bits: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Asymmetric LSQ+ fakequant (learnable scale + learnable zero-point).

    scale: [out, ng] (>0, learnable) ; zp: [out, ng] (learnable real, STE-rounded)
    dequant = (q - z) * s
    """
    out_f, in_f = W.shape
    Qn, Qp = _lsq_qn_qp(q_bits, symmetric=False)
    Wg = _group_reshape(W, group_size)
    s = grad_scale(scale.clamp(min=eps)[..., None], 1.0 / math.sqrt(group_size * max(Qp, 1)))
    z = grad_scale(zp[..., None], 1.0 / math.sqrt(group_size))
    z = round_ste(z).clamp(Qn, Qp)                                 # zero-point rounded at train (STE)
    q = round_ste((Wg / s + z).clamp(Qn, Qp))
    Wq = (q - z) * s
    return _ungroup(Wq, in_f)


@torch.no_grad()
def init_lsq_scale_sym(W: torch.Tensor, group_size: int, q_bits: int) -> torch.Tensor:
    """current_minmax init for symmetric LSQ scale -> [out, ng]."""
    Qp = 2 ** (q_bits - 1) - 1
    Wg = _group_reshape(W, group_size)
    return (Wg.abs().amax(dim=-1) / max(Qp, 1)).clamp(min=1e-8)


@torch.no_grad()
def init_lsq_scale_zp_asym(
    W: torch.Tensor, group_size: int, q_bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    current_minmax init for asymmetric LSQ+ (scale, zp) -> ([out, ng], [out, ng]).

    Matches the affine min/max formula on the asym grid [0, 2^b-1]:
        scale = (wmax - wmin) / Qp ; zp = round(-wmin / scale), clamped to [0, Qp].
    """
    Qp = 2 ** q_bits - 1
    Wg = _group_reshape(W, group_size)                             # [out, ng, gs]
    wmax = Wg.amax(dim=-1)
    wmin = Wg.amin(dim=-1)
    scale = ((wmax - wmin) / max(Qp, 1)).clamp(min=1e-8)
    zp = torch.round(-wmin / scale).clamp(0, Qp)
    return scale, zp


@torch.no_grad()
def lsq_quantize_export_sym(
    W: torch.Tensor, scale: torch.Tensor, group_size: int, q_bits: int
) -> torch.Tensor:
    """Export-time symmetric LSQ quantize with the learned scale -> int W [out, in]."""
    out_f, in_f = W.shape
    Qn, Qp = _lsq_qn_qp(q_bits, symmetric=True)
    Wg = _group_reshape(W, group_size)
    q = (Wg / scale[..., None]).round().clamp(Qn, Qp)
    return _ungroup(q, in_f)


@torch.no_grad()
def lsq_quantize_export_asym(
    W: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor, group_size: int, q_bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Export-time asymmetric LSQ+ quantize -> (int W [out, in], z_int [out, ng]).

    Training rounds zp with round_ste, export with .round() -> same integer value,
    which is what keeps train==export. dequant = (q - z_int) * scale.
    """
    out_f, in_f = W.shape
    Qn, Qp = _lsq_qn_qp(q_bits, symmetric=False)
    z_int = zp.round().clamp(Qn, Qp)
    Wg = _group_reshape(W, group_size)
    q = (Wg / scale[..., None] + z_int[..., None]).round().clamp(Qn, Qp)
    return _ungroup(q, in_f), z_int


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
    def on_train_end(self, model: nn.Module, output_dir: Optional[str] = None):
        """Called at the end of training. Cleanup, unwrap, etc.

        output_dir (when given) is the trainer output dir, so handlers that
        self-register extra params (e.g. LSQ scales) can persist them next to
        the checkpoint before the model is unwrapped/saved.
        """
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

    def on_train_end(self, model, output_dir=None):
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

    def __init__(self, peft_module, group_size, q_bits, symmetric, lora_scaling,
                 where="", enable_lsq=False):
        super().__init__()
        self.m = peft_module
        self.group_size = int(group_size)
        self.symmetric = bool(symmetric)
        self.q_bits = int(q_bits)
        self.q_max = (2 ** (q_bits - 1) - 1) if symmetric else (2 ** q_bits - 1)
        self.lora_scaling = float(lora_scaling)
        self.enable_lsq = bool(enable_lsq)
        self.adapter = list(peft_module.lora_A.keys())[0]
        p = float(getattr(peft_module.lora_dropout[self.adapter], "p", 0.0) or 0.0)
        if p > 0.0:
            warnings.warn(
                f"[FullQAT] {where}: lora_dropout={p:g} > 0. The single-GEMM LR-QAT forward uses "
                "the dropout-free B@A, so it won't match a dropout-applied LoRA forward; set "
                "lora_dropout=0 for full QAT.",
                RuntimeWarning,
            )

        if self.enable_lsq:
            # Register the learnable LSQ scale[, zp] as Parameters of the PEFT MODULE so the HF
            # Trainer optimizer (and trainer.py's dedicated param group) can collect them. Init
            # current_minmax from dequant(base); LoRA B is 0 at init so W_curr ≈ base — the
            # initial grid equals the original min-max grid (no jump at step 0).
            with torch.no_grad():
                W_base = _dequant_bnb_4bit(peft_module.base_layer).float()
            dev = W_base.device
            if self.symmetric:
                s0 = init_lsq_scale_sym(W_base, self.group_size, self.q_bits)
                peft_module.lsq_w_scale = nn.Parameter(s0.to(dev), requires_grad=True)
            else:
                s0, z0 = init_lsq_scale_zp_asym(W_base, self.group_size, self.q_bits)
                peft_module.lsq_w_scale = nn.Parameter(s0.to(dev), requires_grad=True)
                peft_module.lsq_w_zp = nn.Parameter(z0.to(dev), requires_grad=True)
            del W_base

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
        if self.enable_lsq:
            if self.symmetric:
                W_fq = groupwise_lsq_symmetric_fakequant(
                    W_curr, m.lsq_w_scale.float(), self.group_size, self.q_bits)
            else:
                W_fq = groupwise_lsq_asym_fakequant(
                    W_curr, m.lsq_w_scale.float(), m.lsq_w_zp.float(),
                    self.group_size, self.q_bits)
        elif self.symmetric:
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
        self.enable_lsq: bool = False
        # {module_name: peft_module} for LSQ scale/zp save at train end.
        self._lsq_modules: dict = {}

    def prepare_model(self, model, cfg, **kwargs):
        q_bits = cfg["model"]["quant_bits"]
        # Default MUST match export.merge_and_export (also 128). Training fakequant grid must equal
        # the PTQ grid at export time.
        group_size = cfg["qat"].get("group_size", 128)
        symmetric = cfg["qat"].get("symmetric", True)
        lora_scaling = cfg["lora"]["alpha"] / cfg["lora"]["rank"]
        target_modules = set(cfg["lora"]["target_modules"])
        self.enable_lsq = bool(cfg["qat"].get("lsq", {}).get("enabled", False))

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
                lora_scaling=lora_scaling, where=name, enable_lsq=self.enable_lsq,
            ))
            if self.enable_lsq:
                self._lsq_modules[name] = module
            count += 1

        print(
            f"[FullQAT] Installed LR-QAT injectors on {count} PEFT modules "
            f"(group_size={group_size}, bits={q_bits}, symmetric={symmetric}, "
            f"enable_lsq={self.enable_lsq})."
        )
        if self.enable_lsq:
            print(f"[FullQAT] LSQ {'+ (asym, learn scale+zp)' if not symmetric else '(sym, learn scale)'} "
                  f"enabled on {count} modules; scale[,zp] registered as nn.Parameters.")
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

    def save_lsq_scales(self, output_dir: str) -> None:
        """
        Persist learned LSQ scale[, zp] to <output_dir>/lsq_scales.pt as
        {module_name: {"scale": ..., "zp": ...}}. PEFT save_pretrained does NOT
        save these self-registered params, so the export reads them from here.
        Must run BEFORE remove()/save so the params still carry the trained values.
        """
        if not (self.enable_lsq and self._lsq_modules):
            return
        import os
        payload = {}
        for name, module in self._lsq_modules.items():
            entry = {"scale": module.lsq_w_scale.detach().float().cpu().clone()}
            if hasattr(module, "lsq_w_zp"):
                entry["zp"] = module.lsq_w_zp.detach().float().cpu().clone()
            payload[name] = entry
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "lsq_scales.pt")
        torch.save(payload, path)
        print(f"[FullQAT] Saved LSQ scales for {len(payload)} modules to {path}")

    def on_train_end(self, model, output_dir: Optional[str] = None):
        """Remove the LR-QAT hooks so checkpoint saving and export see a clean bnb Linear4bit.

        LSQ scales are saved separately (save_lsq_scales) BEFORE the injectors are removed —
        the QATCallback passes output_dir so the lsq_scales.pt lands next to the checkpoint.
        """
        if output_dir is not None:
            self.save_lsq_scales(output_dir)
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
