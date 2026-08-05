"""
qat_permute_sqat.py — Permuted Selective-QAT: NF4 base + LoRA, fakequant on the salient slice.

This module now owns ONLY the parts that are specific to the LoRA-based Selective-QAT method.
The method-agnostic machinery it is built on was extracted into three shared modules (SALT-Q
reuses all of them):

    src/quant_primitives.py  canonical group-quantization grid (fakequant / quantize / dequantize)
    src/permute_common.py    calibration, saliency, segmentation, P_k / P4_l / Hadamard, boundary
                             gathers, build_permuted_fp16_checkpoint, perm-meta I/O, AWQ scales
    src/gptq.py              GPTQ / OBS (layer + sequential whole-model)

Everything those modules export is RE-EXPORTED here, so existing imports
(`from src.qat_permute_sqat import group_quantize, ...`) keep working unchanged.

What is still implemented here:

  Stage-2 training — the fused Selective-QAT forward:
    Injection is BLOCK-LEVEL and FUSED via hooks (no forward replacement):
      - Attention: a forward_pre_hook on self_attn computes ONE fused delta for q/k/v
        (one BMM for B@A_S when shapes are uniform, one fused fakequant, one GEMM), splits it,
        and stashes the q/k/v slices; forward_hooks on q/k/v_proj add their slice.
      - MLP: gate/up_proj likewise → one fused delta, one GEMM, split + add.
      - down_proj: no sibling to fuse with → a single self-contained forward_hook (small GEMM).
    Net per block: 2 fused big GEMMs (QKV, GateUp) + 1 small GEMM (down) instead of 5 separate
    small delta GEMMs — fewer kernel launches, same QLoRA main path.

Hard invariants:
  * Never materialize the full merged weight W + B@A. Only the salient slice
    W_curr_S = W_base_salient + (B @ A_S) * lora_scaling  of shape [out, layer_group_k].
  * Never replace the BnB/QLoRA projection forward — deltas are ADDED via forward hooks.
  * layer_group_k % group_size == 0. No pre-permutation salient_idx, no index_select/gather on the
    salient slice. X_S = hidden_states[..., :layer_group_k].
  * QKV never share qparams; Gate/Up never share qparams — fakequant is per-output-row, so
    concatenating along the output dim before quantizing is identical to quantizing separately.

Quantize ONCE, not twice: the permute/fold runs on a clean fp16 base which is saved and reloaded
through the standard load_in_4bit path (build_permuted_fp16_checkpoint), so NF4 quantizes the
permuted weights exactly once (no dequant→permute→requant round-trip).

The boundary gather is NOT a weight transform — it is a runtime residual reorder that cannot be
folded offline (the skip connection has no weight carrier). It MUST be re-registered after reload
for training (prepare_model) AND on the exported model for inference (register_boundary_gathers_from_meta).

LoRA dropout: weight-level (W_base + B@A_S) injection equals the runtime LoRA forward only when
lora_dropout == 0. A non-zero dropout triggers a warning (recommend 0 for selective-QAT training).

Stage-1 equivalence verification (P_k + P4 + H all closed, plain fp32) lives in
scripts/verify_permute.py, which drives permute_common.build_and_verify_permutation_fp32().
"""

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.qat_base import QATHandler
from src.qat_base import (
    init_lsq_scale_sym as _lsq_init_sym,
    init_lsq_scale_zp_asym as _lsq_init_asym,
)
from src.qat_sqat import dequantize_layer

# ---------------------------------------------------------------------------
# Re-exports — keep `from src.qat_permute_sqat import X` working for every X that
# used to live in this file. New code should import from the source modules directly.
# ---------------------------------------------------------------------------
from .quant_primitives import (  # noqa: F401
    _asym_q_max,
    _asym_qparams,
    _strip_peft_prefix,
    _sym_q_max,
    _sym_scale,
    collect_lsq_scales_from_model,
    group_dequantize,
    group_fakequant,
    group_quantize,
    groupwise_asymmetric_fakequant,
    groupwise_symmetric_fakequant,
    lsq_scale_for_module,
    round_ste,
    verify_permute_quant_consistency,
)
from .permute_common import (  # noqa: F401
    PERM_META_FILENAME,
    PERMUTE_TARGET_TERMINALS,
    QAT_FAKEQUANT_TERMINALS,
    TARGET_OUTLIER_CAPTURE,
    BoundaryGatherHook,
    _boundary_layer_indices,
    _boundary_offsets,
    _build_segment_perm,
    _collect_second_moments,
    _compute_boundary_perm,
    _down_layer_group_ks,
    _expand_segment_group_ks,
    _layer_idx_from_module_name,
    _residual_outlier_sets,
    _resolve_decoder_layers,
    _resolve_llama_model,
    _round_group_k_to_group_size,
    _segment_group_ks_for_boundaries,
    _segment_outlier_mask,
    _segment_sources,
    _source_outliers,
    _unwrap_perm_model_meta,
    apply_block_internal_permutations_fp32,
    apply_hadamard_rotation_fp32,
    apply_segment_permutation_fp32,
    auto_segment_by_outliers,
    auto_segment_with_fixed_group_k,
    awq_s_for_module,
    build_and_verify_permutation_fp32,
    build_permuted_fp16_checkpoint,
    compute_awq_scales,
    compute_fakequant_param_stats,
    down_layer_group_ks_from_meta,
    format_fakequant_param_stats,
    group_k_for_module_name,
    layer_group_ks_from_meta,
    lm_eval_model_kwargs,
    load_perm_meta,
    maybe_build_gather_aware_hflm,
    register_boundary_gathers,
    register_boundary_gathers_from_meta,
    select_internal_salient_channels,
    select_salient_channels,
    select_salient_channels_variable,
    verify_permutation_equivalence,
)
from .gptq import (  # noqa: F401
    gptq_quantize_layer,
    gptq_quantize_model_sequential,
)


# ============================================================================
# Part 9 — Fused QAT residual (pure math — unit-testable without a real model)
# ============================================================================

def _fused_BA(A_S_list: Sequence[torch.Tensor],
              B_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Concatenated B @ A_S over sibling projections, stacked along the output dim.

    When every (A_S, B) shares the same shape (MHA q/k/v, or gate/up), this is a single batched
    matmul (one kernel). Otherwise (e.g. GQA, where q has more output rows than k/v) it falls
    back to per-projection matmuls concatenated along dim 0 — the fused fakequant and the single
    injection GEMM downstream are unaffected either way.

    Returns: [sum_i out_i, group_k]
    """
    n = len(A_S_list)
    uniform = (
        n > 1
        and all(A.shape == A_S_list[0].shape for A in A_S_list)
        and all(B.shape == B_list[0].shape for B in B_list)
    )
    if uniform:
        Bs = torch.stack(list(B_list), dim=0)              # [n, out, rank]
        As = torch.stack(list(A_S_list), dim=0)            # [n, rank, group_k]
        BA = torch.bmm(Bs, As)                             # [n, out, group_k]
        return BA.reshape(n * BA.shape[1], BA.shape[2])    # rows ordered proj0,proj1,...
    return torch.cat([B @ A for B, A in zip(B_list, A_S_list)], dim=0)


def fused_qat_residual_outputs(
    W_base_salient: torch.Tensor,        # [sum_out, group_k]  (frozen NF4 base salient slice)
    A_S_list: Sequence[torch.Tensor],    # each [rank, group_k]
    B_list: Sequence[torch.Tensor],      # each [out_i, rank]
    out_splits: Sequence[int],           # [out_0, out_1, ...]
    X_S: torch.Tensor,                   # [..., group_k]  (= hidden_states[..., :group_k])
    group_size: int,
    q_bits: int,
    symmetric: bool,
    lora_scaling: float,
    awq_s: Optional[torch.Tensor] = None,   # [group_k] per-input-channel AWQ scale, or None
    lsq_scale: Optional[torch.Tensor] = None,   # [sum_out, n_sal_g] learned LSQ scale, or None
    lsq_zp: Optional[torch.Tensor] = None,      # [sum_out, n_sal_g] learned LSQ zp (asym), or None
) -> List[torch.Tensor]:
    """
    Compute the per-projection QAT residual outputs to be ADDED to each projection's output.

    W_curr = W_base_salient + (B @ A_S) * lora_scaling          [sum_out, group_k]
    delta  = fakequant(W_curr) - W_curr                          (STE residual, original space)
    Y      = F.linear(X_S, delta)                                [..., sum_out]   (ONE GEMM)
    return   Y.split(out_splits, dim=-1)

    AWQ-style per-channel scaling: if `awq_s` (a per-input-channel vector of length group_k, shared
    across the fused siblings) is given, the salient slice is quantized in the AMPLIFIED space —
    the salient input channels' weight columns are scaled up by S before quantization so the
    high-activation channels survive the shared group grid, then divided back out:

        W_fq = fakequant(W_curr * S) ;  delta = W_fq / S - W_curr ;  Y = F.linear(X_S, delta)

    The activation X_S is UNCHANGED — the 1/S is baked into the (dense) weight at export, so this is
    purely a better quantization grid (no runtime activation scaling, output bit-identical to a
    fold-1/S-into-the-preceding-LN deployment). The main NF4+LoRA path is untouched.

    LSQ: if `lsq_scale` (+ `lsq_zp` for asym) is given, the inner fakequant uses the LEARNED scale[,
    zp] on the LSQ grid instead of per-step min-max. It composes with AWQ — the scale/zp are learned
    in the SAME amplified space the fakequant runs in (W_curr*S), so the outer *S … /S is untouched.
    The scale rows are concatenated over the fused siblings in the SAME order as W_base_salient.

    The salient slice is the ONLY weight materialized. Quantization runs in fp32; the residual
    is cast back to X_S.dtype.
    """
    BA     = _fused_BA(A_S_list, B_list).to(torch.float32)         # [sum_out, group_k]
    W_curr = W_base_salient.to(torch.float32) + BA * lora_scaling  # [sum_out, group_k]
    if lsq_scale is not None:
        fixed = lsq_scale.float() if symmetric else (lsq_scale.float(), lsq_zp.float())
    else:
        fixed = None
    if awq_s is not None:
        s     = awq_s.to(torch.float32).view(1, -1)                # [1, group_k]
        W_fq  = group_fakequant(W_curr * s, group_size, q_bits, symmetric, fixed_scale=fixed)
        delta = (W_fq / s - W_curr).to(X_S.dtype)                  # STE residual (original space)
    else:
        W_fq  = group_fakequant(W_curr, group_size, q_bits, symmetric, fixed_scale=fixed)
        delta = (W_fq - W_curr).to(X_S.dtype)
    Y      = F.linear(X_S, delta)                                  # [..., sum_out], one GEMM
    return list(torch.split(Y, list(out_splits), dim=-1))


# ============================================================================
# Part 10 — PEFT projection helpers
# ============================================================================

def _has_lora(proj: nn.Module) -> bool:
    return hasattr(proj, "base_layer") and hasattr(proj, "lora_A") and len(proj.lora_A) > 0


def _adapter_name(proj: nn.Module) -> str:
    return list(proj.lora_A.keys())[0]


def _lora_A_S(proj: nn.Module, group_k: int) -> torch.Tensor:
    """A[:, :group_k]: [rank, group_k] — a view into the global LoRA A (no copy)."""
    return proj.lora_A[_adapter_name(proj)].weight[:, :group_k]


def _lora_B(proj: nn.Module) -> torch.Tensor:
    """B: [out_features, rank]."""
    return proj.lora_B[_adapter_name(proj)].weight


def _lora_dropout_p(proj: nn.Module) -> float:
    if not hasattr(proj, "lora_dropout") or len(proj.lora_dropout) == 0:
        return 0.0
    d = proj.lora_dropout[_adapter_name(proj)]
    return float(getattr(d, "p", 0.0) or 0.0)


def _warn_if_lora_dropout(projs: Sequence[nn.Module], where: str) -> None:
    ps = [_lora_dropout_p(p) for p in projs]
    if any(p > 0.0 for p in ps):
        warnings.warn(
            f"[qat_permute_sqat] {where}: lora_dropout={max(ps):.3g} > 0. Weight-level QAT "
            "injection (W_base + B@A_S) is NOT equivalent to the dropout-applied LoRA forward; "
            "results will be biased. Set lora_dropout=0 for selective-QAT training.",
            RuntimeWarning,
        )


def _dequant_base_salient(proj: nn.Module, group_k: int) -> torch.Tensor:
    """Frozen dequantized NF4 base, salient slice [out, group_k], fp32 on the proj's device."""
    W = dequantize_layer(proj)                              # [out, in]
    return W[:, :group_k].detach().to(torch.float32).contiguous()


# ============================================================================
# Part 11 — Block-level fused injectors (hook-based; original forwards untouched)
# ============================================================================

class _FusedSiblingQATInjector(nn.Module):
    """
    Shared logic for fusing sibling projections that read the SAME block input (q/k/v reading
    the input_layernorm output, or gate/up reading the post_attention_layernorm output).

    A forward_pre_hook on the parent block (`self_attn` / `mlp`) computes the fused residual
    ONCE from the block input and stashes per-projection slices; a forward_hook on each
    projection adds its slice to that projection's output. The parent block's forward and the
    projections' BnB/LoRA forwards are never replaced.
    """

    def __init__(
        self,
        block: nn.Module,
        projs: Sequence[nn.Module],
        group_k: int,
        group_size: int,
        q_bits: int,
        symmetric: bool,
        lora_scaling: float,
        where: str,
        awq_s: Optional[torch.Tensor] = None,
        enable_lsq: bool = False,
    ):
        super().__init__()
        assert group_k % group_size == 0, \
            f"group_k={group_k} must be a multiple of group_size={group_size}"
        self.projs        = list(projs)
        self.group_k      = group_k
        self.group_size   = group_size
        self.q_bits       = q_bits
        self.symmetric    = symmetric
        self.lora_scaling = lora_scaling
        self.where        = where
        self.enable_lsq   = bool(enable_lsq)

        _warn_if_lora_dropout(self.projs, where)

        bases = [_dequant_base_salient(p, group_k) for p in self.projs]
        self.out_splits = tuple(b.shape[0] for b in bases)
        # Frozen fused base salient slice [sum_out, group_k]; not saved to checkpoints.
        self.register_buffer("W_base_salient", torch.cat(bases, dim=0), persistent=False)

        # AWQ-style per-input-channel scale S for the salient slice (shared by the fused siblings,
        # since they read the same block input). Quantizing W_S*S protects the high-activation
        # channels; the /S is baked into the dense weight at export (no runtime activation scaling).
        # On the buffer's device — the injector is not a model submodule, so model.to() won't move it.
        if awq_s is not None:
            self.register_buffer(
                "awq_s",
                awq_s.to(torch.float32).view(-1).to(self.W_base_salient.device),
                persistent=False,
            )
        else:
            self.awq_s = None

        # LSQ: register a learned scale[, zp] PER PROJ (nn.Parameter on the proj module, so the HF
        # Trainer optimizer collects it and export can look it up by the proj's module name). Shape
        # [out_i, n_sal_g]. Init current_minmax from the base salient slice IN THE SAME (amplified)
        # space the fakequant runs in, so the initial grid matches the old min-max grid (no jump).
        if self.enable_lsq:
            n_sal_g = group_k // group_size
            for p, b in zip(self.projs, bases):
                W0 = b.to(torch.float32)                           # [out_i, group_k]
                if self.awq_s is not None:
                    W0 = W0 * self.awq_s.view(1, -1).to(W0.device)
                if symmetric:
                    s0 = _lsq_init_sym(W0, group_size, q_bits)     # [out_i, n_sal_g]
                    p.lsq_w_scale = nn.Parameter(s0.to(W0.device), requires_grad=True)
                else:
                    s0, z0 = _lsq_init_asym(W0, group_size, q_bits)
                    p.lsq_w_scale = nn.Parameter(s0.to(W0.device), requires_grad=True)
                    p.lsq_w_zp = nn.Parameter(z0.to(W0.device), requires_grad=True)
                assert p.lsq_w_scale.shape[1] == n_sal_g

        self._deltas: Optional[List[torch.Tensor]] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._handles.append(block.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        for i, p in enumerate(self.projs):
            self._handles.append(p.register_forward_hook(self._make_add_hook(i)))

    @staticmethod
    def _extract_hidden_states(args, kwargs) -> torch.Tensor:
        if len(args) > 0 and torch.is_tensor(args[0]):
            return args[0]
        if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
            return kwargs["hidden_states"]
        raise RuntimeError(
            "FusedSiblingQATInjector: could not locate hidden_states in the block forward args."
        )

    def _pre_hook(self, module, args, kwargs):
        hidden_states = self._extract_hidden_states(args, kwargs)
        X_S = hidden_states[..., :self.group_k]
        A_S_list = [_lora_A_S(p, self.group_k) for p in self.projs]
        B_list   = [_lora_B(p) for p in self.projs]
        lsq_scale = lsq_zp = None
        if self.enable_lsq:
            # Concatenate the per-proj learned scale[, zp] in the SAME order as W_base_salient.
            lsq_scale = torch.cat([p.lsq_w_scale for p in self.projs], dim=0)
            if not self.symmetric:
                lsq_zp = torch.cat([p.lsq_w_zp for p in self.projs], dim=0)
        self._deltas = fused_qat_residual_outputs(
            self.W_base_salient, A_S_list, B_list, self.out_splits, X_S,
            self.group_size, self.q_bits, self.symmetric, self.lora_scaling,
            awq_s=self.awq_s, lsq_scale=lsq_scale, lsq_zp=lsq_zp,
        )
        return None  # do not modify the block inputs

    def _make_add_hook(self, idx: int):
        def _add(module, inp, out):
            # _deltas is populated by the parent pre-hook earlier in the same forward.
            if self._deltas is None:
                return out
            result = out + self._deltas[idx]
            # Fix C: drop the Python reference to this (activation-sized) delta as
            # soon as it is consumed. The add result + autograd graph keep what
            # backward needs; without this the whole layer's deltas stay pinned on
            # self until the NEXT step's pre-hook overwrites them (~hundreds of MB
            # per layer x num_layers held idle between steps). Clearing per-index is
            # order-independent (does not assume sibling execution order) and the
            # pre-hook repopulates _deltas before these add-hooks run again
            # (including under gradient-checkpoint recompute), so it is safe.
            self._deltas[idx] = None
            return result
        return _add

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._deltas = None


class FusedAttnQATInjector(_FusedSiblingQATInjector):
    """Fuse q_proj/k_proj/v_proj. Handles GQA (q_out != kv_out) via the cat fallback."""

    def __init__(self, attn, q_proj, k_proj, v_proj, **kw):
        super().__init__(attn, [q_proj, k_proj, v_proj], where="self_attn(qkv)", **kw)


class FusedMLPQATInjector(_FusedSiblingQATInjector):
    """Fuse gate_proj/up_proj (always equal output dims = intermediate_size)."""

    def __init__(self, mlp, gate_proj, up_proj, **kw):
        super().__init__(mlp, [gate_proj, up_proj], where="mlp(gate,up)", **kw)


class DownProjQATInjector(nn.Module):
    """
    Single-projection QAT residual for down_proj (no sibling to fuse with). A self-contained
    forward_hook reads the projection input (the act(gate)*up activation, already permuted by P4
    so the salient channels are at [0:layer_group_k]) and adds one small delta GEMM to the output.
    """

    def __init__(
        self,
        down_proj: nn.Module,
        group_k: int,
        group_size: int,
        q_bits: int,
        symmetric: bool,
        lora_scaling: float,
        awq_s: Optional[torch.Tensor] = None,
        enable_lsq: bool = False,
    ):
        super().__init__()
        assert group_k % group_size == 0, \
            f"group_k={group_k} must be a multiple of group_size={group_size}"
        self.down_proj    = down_proj
        self.group_k      = group_k
        self.group_size   = group_size
        self.q_bits       = q_bits
        self.symmetric    = symmetric
        self.lora_scaling = lora_scaling
        self.enable_lsq   = bool(enable_lsq)

        _warn_if_lora_dropout([down_proj], "mlp(down)")
        self.register_buffer(
            "W_base_salient", _dequant_base_salient(down_proj, group_k), persistent=False
        )
        # AWQ scale for down_proj's own salient (P4-permuted) intermediate input slice.
        if awq_s is not None:
            self.register_buffer(
                "awq_s",
                awq_s.to(torch.float32).view(-1).to(self.W_base_salient.device),
                persistent=False,
            )
        else:
            self.awq_s = None
        self.out_splits = (self.W_base_salient.shape[0],)

        # LSQ: learned scale[, zp] for down_proj's salient slice (registered on the proj module).
        if self.enable_lsq:
            W0 = self.W_base_salient.to(torch.float32)
            if self.awq_s is not None:
                W0 = W0 * self.awq_s.view(1, -1).to(W0.device)
            if symmetric:
                s0 = _lsq_init_sym(W0, group_size, q_bits)
                down_proj.lsq_w_scale = nn.Parameter(s0.to(W0.device), requires_grad=True)
            else:
                s0, z0 = _lsq_init_asym(W0, group_size, q_bits)
                down_proj.lsq_w_scale = nn.Parameter(s0.to(W0.device), requires_grad=True)
                down_proj.lsq_w_zp = nn.Parameter(z0.to(W0.device), requires_grad=True)

        self._handles = [down_proj.register_forward_hook(self._hook)]

    def _hook(self, module, inp, out):
        X_S = inp[0][..., :self.group_k]
        A_S = _lora_A_S(self.down_proj, self.group_k)
        B   = _lora_B(self.down_proj)
        lsq_scale = lsq_zp = None
        if self.enable_lsq:
            lsq_scale = self.down_proj.lsq_w_scale
            if not self.symmetric:
                lsq_zp = self.down_proj.lsq_w_zp
        (delta_out,) = fused_qat_residual_outputs(
            self.W_base_salient, [A_S], [B], self.out_splits, X_S,
            self.group_size, self.q_bits, self.symmetric, self.lora_scaling,
            awq_s=self.awq_s, lsq_scale=lsq_scale, lsq_zp=lsq_zp,
        )
        return out + delta_out

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


# ============================================================================
# Part 12 — Install / remove fused injectors
# ============================================================================

def install_fused_selective_qat(
    model: nn.Module,
    group_k: int,
    group_size: int,
    q_bits: int,
    symmetric: bool,
    lora_scaling: float,
    target_modules: Sequence[str],
    include_down_proj: bool = True,
    awq_scales: Optional[dict] = None,
    layer_group_ks: Optional[Sequence[int]] = None,
    down_layer_group_ks: Optional[Sequence[int]] = None,
    enable_lsq: bool = False,
) -> List[nn.Module]:
    """
    Install block-level fused Selective-QAT injectors on every decoder layer.

    - q/k/v_proj   → one FusedAttnQATInjector per layer (pre-hook on self_attn + add-hooks).
    - gate/up_proj → one FusedMLPQATInjector per layer.
    - down_proj    → one DownProjQATInjector per layer (single small GEMM, no fusion).

    awq_scales: if given ({"attn"/"mlp"/"down": [num_layers, max_group_k]}), the salient slice of each
    projection group is quantized in the AWQ-amplified space (q/k/v share the layer's "attn" S,
    gate/up share "mlp", down_proj uses "down"). o_proj is never injected (per-head Hadamard only).
    The /S is baked into the dense weight at export — a better quant grid, bit-identical output.

    Only projections that are LoRA-wrapped AND listed in `target_modules` are injected.
    Returns the list of injectors (keep a reference; call .remove() on each to uninstall).
    """
    layers = list(_resolve_decoder_layers(model))
    if layer_group_ks is None:
        layer_group_ks = [int(group_k)] * len(layers)
    else:
        layer_group_ks = [int(x) for x in layer_group_ks]
        assert len(layer_group_ks) == len(layers), (
            f"len(layer_group_ks)={len(layer_group_ks)} != num_layers={len(layers)}"
        )
    if down_layer_group_ks is None:
        down_layer_group_ks = list(layer_group_ks)
    else:
        down_layer_group_ks = [int(x) for x in down_layer_group_ks]
        assert len(down_layer_group_ks) == len(layers), (
            f"len(down_layer_group_ks)={len(down_layer_group_ks)} != num_layers={len(layers)}"
        )

    def _s(src: str, l: int, gk: int) -> Optional[torch.Tensor]:
        if not awq_scales or src not in awq_scales:
            return None
        return torch.as_tensor(awq_scales[src][l][:gk], dtype=torch.float32)

    tset = set(target_modules)
    injectors: List[nn.Module] = []

    for l, layer in enumerate(layers):
        attn = layer.self_attn
        mlp  = layer.mlp
        residual_gk_l = int(layer_group_ks[l])
        down_gk_l = int(down_layer_group_ks[l])
        common_residual = dict(
            group_k=residual_gk_l, group_size=group_size, q_bits=q_bits,
            symmetric=symmetric, lora_scaling=lora_scaling, enable_lsq=enable_lsq,
        )
        common_down = {**common_residual, "group_k": down_gk_l}

        if {"q_proj", "k_proj", "v_proj"} <= tset and all(
            _has_lora(getattr(attn, n)) for n in ("q_proj", "k_proj", "v_proj")
        ):
            injectors.append(FusedAttnQATInjector(
                attn, attn.q_proj, attn.k_proj, attn.v_proj,
                awq_s=_s("attn", l, residual_gk_l), **common_residual
            ))

        if {"gate_proj", "up_proj"} <= tset and all(
            _has_lora(getattr(mlp, n)) for n in ("gate_proj", "up_proj")
        ):
            injectors.append(FusedMLPQATInjector(
                mlp, mlp.gate_proj, mlp.up_proj,
                awq_s=_s("mlp", l, residual_gk_l), **common_residual
            ))

        if include_down_proj and "down_proj" in tset and _has_lora(mlp.down_proj):
            injectors.append(DownProjQATInjector(
                mlp.down_proj, awq_s=_s("down", l, down_gk_l), **common_down,
            ))

    print(
        f"[qat_permute_sqat] Installed fused Selective-QAT injectors: "
        f"{sum(isinstance(i, FusedAttnQATInjector) for i in injectors)} attn, "
        f"{sum(isinstance(i, FusedMLPQATInjector) for i in injectors)} mlp, "
        f"{sum(isinstance(i, DownProjQATInjector) for i in injectors)} down  "
        f"(residual_group_k_by_layer={min(layer_group_ks)}..{max(layer_group_ks)}, "
        f"down_group_k_by_layer={min(down_layer_group_ks)}..{max(down_layer_group_ks)}, "
        f"group_size={group_size}, symmetric={symmetric}, "
        f"awq_scale={'on' if awq_scales else 'off'}, enable_lsq={enable_lsq})"
    )
    return injectors


def remove_fused_selective_qat(injectors: Sequence[nn.Module]) -> None:
    """Remove all hooks installed by install_fused_selective_qat."""
    for inj in injectors:
        inj.remove()


# ============================================================================
# Part 14 — QATHandler
# ============================================================================

class SegmentPermutedSelectiveQAT(QATHandler):
    """
    Segment-Shared Permutation QAT handler (Stage 2).

    The permute/fold happens BEFORE NF4 quantization, on a clean fp16 base, inside
    build_permuted_fp16_checkpoint() (called by scripts/train.py on rank 0). That base is saved
    and reloaded through the standard load_in_4bit path, so NF4 quantizes the permuted weights
    exactly once (no dequant→permute→requant round-trip).

    prepare_model() therefore receives the ALREADY-permuted, freshly-NF4-quantized PEFT model
    plus `perm_meta`, and only:
      1. Re-registers the runtime boundary gathers (residual reorder; cannot be folded).
      2. [stage>=2] Installs block-level fused Selective-QAT injectors (hooks) on
         q/k/v/gate/up/down_proj — QKV and Gate/Up each fuse into ONE big GEMM, down_proj is one
         small GEMM. o_proj is NOT injected (Hadamard-rotated only).
      3. Attaches _sqat_permute_meta for export.
    """

    def __init__(self):
        self.injectors:            List[nn.Module] = []
        self.boundary_hooks:       List[BoundaryGatherHook] = []
        self.boundary_perms:       List[torch.LongTensor] = []
        self.segment_perms:        Dict[int, List[int]] = {}
        self.block_internal_perms: Dict[str, List[int]] = {}
        self.enable_lsq:           bool = False
        self._lsq_proj_names:      Dict[int, Tuple[str, nn.Module]] = {}

    def prepare_model(
        self,
        model: nn.Module,
        cfg: dict,
        tokenizer=None,
        perm_meta: Optional[dict] = None,
        calibration_dataloader: Optional[DataLoader] = None,
        **kwargs,
    ) -> nn.Module:
        assert perm_meta is not None, (
            "sqat_permute prepare_model requires perm_meta from "
            "build_permuted_fp16_checkpoint(). The permute/fold happens in fp16 BEFORE NF4 "
            "(see scripts/train.py), not in this handler."
        )

        sp_cfg         = cfg["qat"]["sqat_permute"]
        stage          = sp_cfg.get("stage", 2)
        group_k        = perm_meta["group_k"]
        group_size     = perm_meta["group_size"]
        layer_group_ks = layer_group_ks_from_meta(perm_meta) or [int(group_k)] * int(sum(perm_meta["boundary_sizes"]))
        down_layer_group_ks = down_layer_group_ks_from_meta(perm_meta) or list(layer_group_ks)
        q_bits         = cfg["model"]["quant_bits"]
        symmetric      = cfg["qat"].get("symmetric", True)
        awq_enabled    = bool((sp_cfg.get("awq_scale", {}) or {}).get("enabled", False))
        enable_lsq     = bool(cfg["qat"].get("lsq", {}).get("enabled", False))
        lora_scaling   = cfg["lora"]["alpha"] / cfg["lora"]["rank"]
        target_modules = cfg["lora"]["target_modules"]
        d_model        = perm_meta["d_model"]
        awq_scales     = perm_meta.get("awq_scales") if awq_enabled else None

        # ---- 1) Runtime boundary gathers (residual reorder P_k → P_{k+1}). ----
        # MUST exist for every forward — at training (here) AND inference (eval scripts call
        # register_boundary_gathers_from_meta). Skip connections carry no weight, so this cannot
        # be folded offline; it is the one unavoidable runtime cost.
        self.boundary_perms = [
            torch.as_tensor(bp, dtype=torch.long) for bp in perm_meta["boundary_perms"]
        ]
        bli                       = perm_meta["boundary_layer_indices"]
        self.segment_perms        = perm_meta.get("segment_perms", {})
        self.block_internal_perms = perm_meta.get("block_internal_perms", {})
        self.boundary_hooks = register_boundary_gathers(
            model, self.boundary_perms, bli, d_model
        )

        # ---- 2) Stage 2 — install block-level fused Selective-QAT injectors. ----
        self.enable_lsq = enable_lsq
        if stage >= 2:
            self.injectors = install_fused_selective_qat(
                model,
                group_k=group_k,
                group_size=group_size,
                q_bits=q_bits,
                symmetric=symmetric,
                lora_scaling=lora_scaling,
                target_modules=target_modules,
                include_down_proj=True,
                awq_scales=awq_scales,
                layer_group_ks=layer_group_ks,
                down_layer_group_ks=down_layer_group_ks,
                enable_lsq=enable_lsq,
            )

        # ---- 3) Attach export metadata (perm_meta + q_bits + symmetric + awq flag). ----
        model._sqat_permute_meta = {
            **perm_meta, "q_bits": q_bits, "symmetric": symmetric,
            "awq_scale": awq_enabled, "lsq": enable_lsq,
        }
        # Keep a reference so collect (after training) can read the learned scales off the live
        # proj modules. The proj→name map lets us key lsq_scales by the export (stripped) name.
        self._lsq_proj_names = {}
        if enable_lsq:
            for name, mod in model.named_modules():
                if hasattr(mod, "lsq_w_scale"):
                    self._lsq_proj_names[id(mod)] = (name, mod)

        print(f"[SegPerm] prepare_model done: stage={stage}, "
              f"num_runtime_permutes={len(self.boundary_perms)}, "
              f"residual_group_k_by_layer={min(layer_group_ks)}..{max(layer_group_ks)}, "
              f"down_group_k_by_layer={min(down_layer_group_ks)}..{max(down_layer_group_ks)} "
              f"(max={group_k}), group_size={group_size}, symmetric={symmetric}, "
              f"awq_scale={awq_enabled}, enable_lsq={enable_lsq}")
        if perm_meta.get("fakequant_param_stats"):
            print(
                "[SegPerm] Selective-QAT fakequant parameter coverage: "
                f"{format_fakequant_param_stats(perm_meta['fakequant_param_stats'])}"
            )
        return model

    def on_train_begin(self, model: nn.Module): pass
    def on_step_end(self, model: nn.Module, step: int): pass
    def on_train_end(self, model: nn.Module, output_dir=None): pass
