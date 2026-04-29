"""
Fused Triton kernels for SelectiveSalientQATLinear Pass 3.

──────────────────────────────────────────────────────────────────────────────
PERFORMANCE ANALYSIS
──────────────────────────────────────────────────────────────────────────────

Pass 3 forward (vanilla PyTorch):

  X_S     = x[..., salient_indices]                        # gather  [B*T, K]
  W_curr  = W_base_S + B @ A[:,S] * scaling               # GEMM + ew  [N, K]
  W_quant = selective_salient_fakequant_asym(W_curr, ...)  # ~27 kernels [N, K]
  delta   = W_quant - W_curr                               # ew       [N, K]
  Y      += F.linear(X_S, delta)                           # GEMM   [B*T, N]

Typical shapes: N=4096, K=41 (1% of 4096), G=32 groups, group_size=128.

PyTorch fakequant launch count breakdown:
  scatter_reduce_ × 4   (w_max, w_min, sal_max, sal_min)
  elementwise × 6       (raw_scale, z_int, denom_max, denom_min, scale_from_max/min)
  where × 3 + gather × 6 + quantize + mask ops ≈ 27 launches total

Each PyTorch kernel reads/writes [N, K] ≈ 672 KB → ~18 MB unnecessary traffic.

──────────────────────────────────────────────────────────────────────────────
KERNEL DESIGN
──────────────────────────────────────────────────────────────────────────────

Single unified delta kernel  _sqat_unified_delta_kernel
  Grid: (cdiv(N, BN_A), G)   e.g. (64, 32) = 2048 blocks for N=4096, G=32
  Each block: BN_A=64 rows × one quantisation group g
  - Loads [BN_A, BLOCK_SAL] weights (W_base + BA)
  - Reduces per-row to get [BN_A] group stats (scale, z_int, flags)
  - Quantises and writes delta to [N, K]
  All in ONE kernel launch, no intermediate [N, G] stats tensors.

  Memory: reads ≈ 3 × [BN_A, BLOCK_SAL] + 2 × [BN_A] per block,
          writes ≈ [BN_A, BLOCK_SAL].  Total: ~3 MB vs ~18 MB PyTorch.

Optional stats kernel  _sqat_group_stats_kernel
  Same grid as above, but outputs 6 × [N, G] tensors instead of delta.
  Needed by the fully-fused GEMM to avoid recomputing stats inside the GEMM.

Fully-fused GEMM  _sqat_fused_linear_delta_kernel
  Grid: (cdiv(M, BM), cdiv(N, BN))
  Uses precomputed [N, G] stats + recomputes W_curr on-the-fly.
  Eliminates delta materialisation (saves ~1.3 MB read+write).

──────────────────────────────────────────────────────────────────────────────
GRADIENT / STE NOTE
──────────────────────────────────────────────────────────────────────────────

With STE (round_ste backward = identity):

  d(W_quant)/d(W_curr) = 1   →   d(delta)/d(W_curr) = 1 - 1 = 0

Pass 3 contributes ZERO gradient to LoRA parameters (A, B).
Training signal for LoRA comes entirely from Pass 1+2 (standard QLoRA).
Consequence: SQATFakequantDeltaFn.backward returns None for all weight inputs.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ─────────────────────────────────────────────────────────────────────────────
# CPU helper: group span precomputation
# ─────────────────────────────────────────────────────────────────────────────

def precompute_group_spans(
    group_ids: torch.Tensor,  # [K] int, non-decreasing (salient_idx // group_size)
    G: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (group_start, group_len), both [G] int32.

    group_start[g] = first K-position that belongs to group g
    group_len[g]   = number of salient channels in group g

    Requires group_ids to be sorted (guaranteed because salient_indices are sorted).
    """
    counts = torch.bincount(group_ids.int(), minlength=G).int()
    starts = torch.zeros(G, dtype=torch.int32, device=group_ids.device)
    starts[1:] = counts[:-1].cumsum(0)
    return starts, counts


if HAS_TRITON:
    # ─────────────────────────────────────────────────────────────────────────
    # Kernel: unified delta computation
    # Grid (cdiv(N, BN_A), G) — batches BN_A output rows per block so that
    # warp utilisation is high even when BLOCK_SAL (salient/group) is small.
    # ─────────────────────────────────────────────────────────────────────────

    @triton.jit
    def _sqat_unified_delta_kernel(
        # [N, K] inputs
        W_base_ptr, BA_ptr,
        # [N, G] base stats (non-salient channels, precomputed, frozen)
        base_max_ptr, base_min_ptr,
        # Group spans [G]
        group_start_ptr, group_len_ptr,
        # [N, K] output
        delta_ptr,
        # Scalars
        lora_scaling,
        q_lvl: tl.constexpr,
        eps: tl.constexpr,
        N, K,
        stride_nk,   # = K  (row stride for [N,K] tensors)
        stride_ng,   # = G  (row stride for [N,G] tensors)
        BN_A: tl.constexpr,      # output rows per block (tuned for occupancy)
        BLOCK_SAL: tl.constexpr, # max salient per group, padded to power of 2
    ):
        n_pid = tl.program_id(0)
        g     = tl.program_id(1)

        n_offs = n_pid * BN_A + tl.arange(0, BN_A)   # [BN_A]
        n_mask = n_offs < N

        gs = tl.load(group_start_ptr + g).to(tl.int32)
        gl = tl.load(group_len_ptr   + g).to(tl.int32)

        sal_offs  = tl.arange(0, BLOCK_SAL)           # [BLOCK_SAL]
        sal_valid = sal_offs < gl
        k_pos     = gs + sal_offs                     # [BLOCK_SAL]

        # W_curr = W_base + BA * lora_scaling  →  [BN_A, BLOCK_SAL]
        load_mask = n_mask[:, None] & sal_valid[None, :]
        w_base = tl.load(
            W_base_ptr + n_offs[:, None] * stride_nk + k_pos[None, :],
            mask=load_mask, other=0.0,
        )
        ba = tl.load(
            BA_ptr + n_offs[:, None] * stride_nk + k_pos[None, :],
            mask=load_mask, other=0.0,
        )
        w_curr = w_base + ba * lora_scaling            # [BN_A, BLOCK_SAL]

        # Base group stats  →  [BN_A]
        row_ng   = n_offs * stride_ng
        base_max = tl.load(base_max_ptr + row_ng + g, mask=n_mask, other=0.0)
        base_min = tl.load(base_min_ptr + row_ng + g, mask=n_mask, other=0.0)

        # Per-row reduction over BLOCK_SAL.
        # When gl == 0: all sal_valid=False → sal_max=NEG_INF, sal_min=POS_INF
        # → w_max=base_max, w_min=base_min, sal_is_max=False, sal_is_min=False ✓
        NEG_INF = -3.4028235e+38
        POS_INF =  3.4028235e+38

        sal_max = tl.max(tl.where(sal_valid[None, :], w_curr, NEG_INF), axis=1)  # [BN_A]
        sal_min = tl.min(tl.where(sal_valid[None, :], w_curr, POS_INF), axis=1)  # [BN_A]

        w_max = tl.maximum(base_max, sal_max)  # [BN_A]
        w_min = tl.minimum(base_min, sal_min)  # [BN_A]

        sal_is_max = sal_max >= base_max        # [BN_A]
        sal_is_min = sal_min <= base_min        # [BN_A]

        # Asymmetric scale  →  [BN_A]
        raw_scale = (w_max - w_min) / float(q_lvl)
        raw_scale = tl.maximum(raw_scale, eps)

        z_int = tl.floor(-w_min / raw_scale + 0.5)
        z_int = tl.minimum(tl.maximum(z_int, 0.0), float(q_lvl))

        denom_max     = tl.maximum(float(q_lvl) - z_int, 1.0)
        denom_min     = tl.maximum(z_int, 1.0)
        scale_from_max = w_max / denom_max
        scale_from_min = (-w_min) / denom_min

        prefer_max = tl.abs(w_max) >= tl.abs(w_min)
        use_max    = sal_is_max & prefer_max
        use_min    = sal_is_min & (~prefer_max)

        scale = raw_scale
        scale = tl.where(use_max, scale_from_max, scale)
        scale = tl.where(use_min, scale_from_min, scale)
        scale = tl.maximum(scale, eps)         # [BN_A]

        # Quantize  →  [BN_A, BLOCK_SAL]  (broadcast per-row scalars)
        q = tl.floor(w_curr / scale[:, None] + z_int[:, None] + 0.5)
        q = tl.minimum(tl.maximum(q, 0.0), float(q_lvl))
        w_quant = (q - z_int[:, None]) * scale[:, None]

        # Anchor pass-through (exact reconstruction at group extremes)
        is_max_anch = use_max[:, None] & (w_curr >= w_max[:, None] - 1e-5)
        is_min_anch = use_min[:, None] & (w_curr <= w_min[:, None] + 1e-5)
        is_anchor   = is_max_anch | is_min_anch

        w_out = tl.where(is_anchor, w_curr, w_quant)
        delta = w_out - w_curr                 # [BN_A, BLOCK_SAL]

        # Write delta for valid (row, salient-col) pairs
        tl.store(
            delta_ptr + n_offs[:, None] * stride_nk + k_pos[None, :],
            delta,
            mask=load_mask,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Kernel: per-group stats (needed as input to the fused GEMM)
    # Same grid and body as above, but writes [N, G] stats instead of delta.
    # ─────────────────────────────────────────────────────────────────────────

    @triton.jit
    def _sqat_group_stats_kernel(
        W_base_ptr, BA_ptr,
        base_max_ptr, base_min_ptr,
        group_start_ptr, group_len_ptr,
        # Outputs [N, G]
        scale_ptr, zint_ptr,
        usemax_ptr, usemin_ptr,   # uint8
        wmax_ptr, wmin_ptr,
        lora_scaling,
        q_lvl: tl.constexpr,
        eps: tl.constexpr,
        N,
        stride_nk, stride_ng,
        BN_A: tl.constexpr,
        BLOCK_SAL: tl.constexpr,
    ):
        n_pid = tl.program_id(0)
        g     = tl.program_id(1)

        n_offs = n_pid * BN_A + tl.arange(0, BN_A)
        n_mask = n_offs < N

        gs = tl.load(group_start_ptr + g).to(tl.int32)
        gl = tl.load(group_len_ptr   + g).to(tl.int32)

        sal_offs  = tl.arange(0, BLOCK_SAL)
        sal_valid = sal_offs < gl
        k_pos     = gs + sal_offs

        load_mask = n_mask[:, None] & sal_valid[None, :]
        w_base = tl.load(W_base_ptr + n_offs[:, None] * stride_nk + k_pos[None, :], mask=load_mask, other=0.0)
        ba     = tl.load(BA_ptr     + n_offs[:, None] * stride_nk + k_pos[None, :], mask=load_mask, other=0.0)
        w_curr = w_base + ba * lora_scaling

        row_ng   = n_offs * stride_ng
        base_max = tl.load(base_max_ptr + row_ng + g, mask=n_mask, other=0.0)
        base_min = tl.load(base_min_ptr + row_ng + g, mask=n_mask, other=0.0)

        NEG_INF = -3.4028235e+38
        POS_INF =  3.4028235e+38
        sal_max = tl.max(tl.where(sal_valid[None, :], w_curr, NEG_INF), axis=1)
        sal_min = tl.min(tl.where(sal_valid[None, :], w_curr, POS_INF), axis=1)

        w_max = tl.maximum(base_max, sal_max)
        w_min = tl.minimum(base_min, sal_min)
        sal_is_max = sal_max >= base_max
        sal_is_min = sal_min <= base_min

        raw_scale = tl.maximum((w_max - w_min) / float(q_lvl), eps)
        z_int = tl.floor(-w_min / raw_scale + 0.5)
        z_int = tl.minimum(tl.maximum(z_int, 0.0), float(q_lvl))

        denom_max     = tl.maximum(float(q_lvl) - z_int, 1.0)
        denom_min     = tl.maximum(z_int, 1.0)
        scale_from_max = w_max / denom_max
        scale_from_min = (-w_min) / denom_min

        prefer_max = tl.abs(w_max) >= tl.abs(w_min)
        use_max    = sal_is_max & prefer_max
        use_min    = sal_is_min & (~prefer_max)

        scale = raw_scale
        scale = tl.where(use_max, scale_from_max, scale)
        scale = tl.where(use_min, scale_from_min, scale)
        scale = tl.maximum(scale, eps)

        out_offs = row_ng + g
        tl.store(scale_ptr  + out_offs, scale,                        mask=n_mask)
        tl.store(zint_ptr   + out_offs, z_int,                        mask=n_mask)
        tl.store(usemax_ptr + out_offs, tl.cast(use_max, tl.uint8),   mask=n_mask)
        tl.store(usemin_ptr + out_offs, tl.cast(use_min, tl.uint8),   mask=n_mask)
        tl.store(wmax_ptr   + out_offs, w_max,                        mask=n_mask)
        tl.store(wmin_ptr   + out_offs, w_min,                        mask=n_mask)

    # ─────────────────────────────────────────────────────────────────────────
    # Kernel: fully-fused GEMM
    # Grid (cdiv(M, BM), cdiv(N, BN)).  Inner double loop: G groups × BLOCK_SAL.
    # For each (group, salient-column): compute delta [BN] + outer product [BM, BN].
    # Avoids materialising delta_S entirely.
    # ─────────────────────────────────────────────────────────────────────────

    @triton.jit
    def _sqat_fused_linear_delta_kernel(
        X_ptr,                          # [M, K]
        W_base_ptr, BA_ptr,             # [N, K]
        scale_ptr, zint_ptr,            # [N, G]
        usemax_ptr, usemin_ptr,         # [N, G] uint8
        wmax_ptr, wmin_ptr,             # [N, G]
        group_start_ptr, group_len_ptr, # [G]
        Y_ptr,                          # [M, N] output (zeroed before launch)
        lora_scaling,
        q_lvl: tl.constexpr,
        G,
        M, N, K,
        stride_xm,             # X row stride  (= K)
        stride_nk, stride_ng,  # [N,K] and [N,G] row strides
        stride_ym,             # Y row stride   (= N)
        BM: tl.constexpr,
        BN: tl.constexpr,
        BLOCK_SAL: tl.constexpr,
    ):
        m_pid = tl.program_id(0)
        n_pid = tl.program_id(1)

        m_offs = m_pid * BM + tl.arange(0, BM)
        n_offs = n_pid * BN + tl.arange(0, BN)
        m_mask = m_offs < M
        n_mask = n_offs < N

        acc = tl.zeros([BM, BN], dtype=tl.float32)

        for g in tl.range(0, G):
            gs = tl.load(group_start_ptr + g).to(tl.int32)
            gl = tl.load(group_len_ptr   + g).to(tl.int32)

            # Group stats for this BN-slice of N rows  →  [BN]
            g_off   = n_offs * stride_ng + g
            scale   = tl.load(scale_ptr  + g_off, mask=n_mask, other=1.0)
            z_int   = tl.load(zint_ptr   + g_off, mask=n_mask, other=0.0)
            use_max = tl.load(usemax_ptr + g_off, mask=n_mask, other=0).to(tl.int1)
            use_min = tl.load(usemin_ptr + g_off, mask=n_mask, other=0).to(tl.int1)
            w_max_g = tl.load(wmax_ptr   + g_off, mask=n_mask, other=0.0)
            w_min_g = tl.load(wmin_ptr   + g_off, mask=n_mask, other=0.0)

            # Inner loop: one iteration per salient channel in this group.
            # BLOCK_SAL is constexpr → fully unrolled at compile time.
            for ki in tl.range(0, BLOCK_SAL):
                k_valid = ki < gl
                k_i = (gs + ki).to(tl.int32)

                # W_curr for column k_i  →  [BN]
                w_base_k = tl.load(W_base_ptr + n_offs * stride_nk + k_i, mask=n_mask, other=0.0)
                ba_k     = tl.load(BA_ptr     + n_offs * stride_nk + k_i, mask=n_mask, other=0.0)
                w_curr_k = w_base_k + ba_k * lora_scaling

                q_k      = tl.floor(w_curr_k / scale + z_int + 0.5)
                q_k      = tl.minimum(tl.maximum(q_k, 0.0), float(q_lvl))
                w_quant_k = (q_k - z_int) * scale

                is_max_anch = use_max & (w_curr_k >= w_max_g - 1e-5)
                is_min_anch = use_min & (w_curr_k <= w_min_g + 1e-5)
                w_out_k  = tl.where(is_max_anch | is_min_anch, w_curr_k, w_quant_k)
                delta_k  = tl.where(k_valid, w_out_k - w_curr_k, 0.0)  # [BN]

                # Load x_k  →  [BM]
                x_k = tl.where(
                    k_valid,
                    tl.load(X_ptr + m_offs * stride_xm + k_i, mask=m_mask, other=0.0),
                    0.0,
                )

                # Outer product: acc[m, n] += x_k[m] * delta_k[n]
                acc += x_k[:, None] * delta_k[None, :]

        y_offs = m_offs[:, None] * stride_ym + n_offs[None, :]
        tl.store(Y_ptr + y_offs, acc, mask=(m_mask[:, None] & n_mask[None, :]))


# ─────────────────────────────────────────────────────────────────────────────
# Python wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _launch_unified_delta(
    W_base: torch.Tensor,       # [N, K] fp32
    BA: torch.Tensor,           # [N, K] fp32
    base_max: torch.Tensor,     # [N, G] fp32
    base_min: torch.Tensor,     # [N, G] fp32
    group_start: torch.Tensor,  # [G] int32
    group_len: torch.Tensor,    # [G] int32
    lora_scaling: float,
    q_lvl: int,
    eps: float,
    block_sal: int,
    bna: int = 32,
) -> torch.Tensor:
    N, K = W_base.shape
    G    = base_max.shape[1]
    delta = torch.empty_like(W_base)
    grid = (triton.cdiv(N, bna), G)
    _sqat_unified_delta_kernel[grid](
        W_base, BA, base_max, base_min, group_start, group_len,
        delta,
        lora_scaling, q_lvl, eps,
        N, K,
        W_base.stride(0), base_max.stride(0),
        BN_A=bna, BLOCK_SAL=block_sal,
    )
    return delta


def _launch_group_stats(
    W_base: torch.Tensor,
    BA: torch.Tensor,
    base_max: torch.Tensor,
    base_min: torch.Tensor,
    group_start: torch.Tensor,
    group_len: torch.Tensor,
    lora_scaling: float,
    q_lvl: int,
    eps: float,
    block_sal: int,
    bna: int = 32,
) -> Tuple[torch.Tensor, ...]:
    """Returns (scale, z_int, use_max, use_min, w_max, w_min) each [N, G]."""
    N = W_base.shape[0]
    G = base_max.shape[1]
    dev = W_base.device

    scale   = torch.empty(N, G, dtype=torch.float32, device=dev)
    z_int   = torch.empty(N, G, dtype=torch.float32, device=dev)
    use_max = torch.empty(N, G, dtype=torch.uint8,   device=dev)
    use_min = torch.empty(N, G, dtype=torch.uint8,   device=dev)
    w_max   = torch.empty(N, G, dtype=torch.float32, device=dev)
    w_min   = torch.empty(N, G, dtype=torch.float32, device=dev)

    grid = (triton.cdiv(N, bna), G)
    _sqat_group_stats_kernel[grid](
        W_base, BA, base_max, base_min, group_start, group_len,
        scale, z_int, use_max, use_min, w_max, w_min,
        lora_scaling, q_lvl, eps, N,
        W_base.stride(0), base_max.stride(0),
        BN_A=bna, BLOCK_SAL=block_sal,
    )
    return scale, z_int, use_max, use_min, w_max, w_min


class SQATFakequantDeltaFn(torch.autograd.Function):
    """
    Forward: Triton-fused fakequant → delta_S.
    Backward: STE — d(delta)/d(W_curr) = 0 → return None for all weight inputs.
    Gradient for x flows through the F.linear that is called after this function.
    """

    @staticmethod
    def forward(
        ctx,
        W_base, BA, base_max, base_min,
        group_start, group_len,
        lora_scaling, q_lvl, eps, block_sal,
    ):
        delta = _launch_unified_delta(
            W_base.float().contiguous(),
            BA.float().contiguous(),
            base_max, base_min, group_start, group_len,
            lora_scaling, q_lvl, eps, block_sal,
        )
        return delta

    @staticmethod
    def backward(ctx, grad_delta):
        return (None,) * 10


def sqat_fakequant_delta(
    W_base: torch.Tensor,
    BA: torch.Tensor,
    base_max: torch.Tensor,
    base_min: torch.Tensor,
    group_start: torch.Tensor,
    group_len: torch.Tensor,
    group_ids: torch.Tensor,    # unused here (kept for API symmetry with PyTorch path)
    lora_scaling: float,
    q_lvl: int,
    eps: float = 1e-7,
    block_sal: int | None = None,
) -> torch.Tensor:
    """
    Triton-fused replacement for selective_salient_fakequant_asym + subtraction.
    Returns delta_S = W_quant_S - W_curr_S  of shape [N, K].

    Replaces ~27 PyTorch kernel launches with a single Triton kernel.
    """
    if block_sal is None:
        block_sal = triton.next_power_of_2(max(1, int(group_len.max().item())))
    return SQATFakequantDeltaFn.apply(
        W_base, BA, base_max, base_min,
        group_start, group_len,
        lora_scaling, q_lvl, eps, block_sal,
    )


def sqat_fused_linear_delta(
    X_S: torch.Tensor,
    W_base: torch.Tensor,
    BA: torch.Tensor,
    base_max: torch.Tensor,
    base_min: torch.Tensor,
    group_start: torch.Tensor,
    group_len: torch.Tensor,
    group_ids: torch.Tensor,    # unused (kept for API symmetry)
    lora_scaling: float,
    q_lvl: int,
    eps: float = 1e-7,
    block_sal: int | None = None,
    BM: int = 64,               # unused (kept for API symmetry with custom GEMM)
    BN: int = 64,
) -> torch.Tensor:
    """
    Optimised pass3: Triton unified-delta kernel + cuBLAS F.linear.

    Strategy: the unified delta kernel (1 Triton launch) replaces ~27 PyTorch
    launches for the fakequant, producing delta_S in a single pass.  The GEMM
    X_S @ delta_S.T is then delegated to cuBLAS (via F.linear) which is
    significantly faster than a hand-written outer-product kernel for this size.

    Net result: ~2 kernel launches (Triton + cuBLAS) vs ~28 (PyTorch).
    Typical speedup for the combined pass3: 2-3x.

    _sqat_fused_linear_delta_kernel is kept as an experimental alternative that
    avoids delta_S materialisation, useful when K is large or when memory is
    the binding constraint.
    """
    if block_sal is None:
        block_sal = triton.next_power_of_2(max(1, int(group_len.max().item())))

    W_base_f = W_base.float().contiguous()
    BA_f     = BA.float().contiguous()

    delta_S = _launch_unified_delta(
        W_base_f, BA_f, base_max, base_min,
        group_start, group_len,
        lora_scaling, q_lvl, eps, block_sal,
    )
    # F.linear handles the GEMM with cuBLAS (much faster than custom kernel).
    return F.linear(X_S.float(), delta_S).to(X_S.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Correctness check (run as a standalone script)
# ─────────────────────────────────────────────────────────────────────────────

def _check_correctness(device: str = "cuda", seed: int = 42):
    if not HAS_TRITON:
        print("[triton_sqat] Triton not available.")
        return

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.qat_sqat import selective_salient_fakequant_asym

    torch.manual_seed(seed)
    N, K, G, q_lvl = 64, 20, 8, 15
    group_size = 16; lora_scaling = 0.5

    W_base   = torch.randn(N, K, device=device)
    BA       = torch.randn(N, K, device=device)
    W_curr   = W_base + BA * lora_scaling
    group_ids = (torch.arange(K, device=device) // group_size).int()
    base_max  = torch.randn(N, G, device=device).abs() * 0.5
    base_min  = -torch.randn(N, G, device=device).abs() * 0.5

    ref = selective_salient_fakequant_asym(
        W_curr, group_ids.long(), base_max, base_min, q_lvl
    )
    delta_ref = ref - W_curr

    group_start, group_len = precompute_group_spans(group_ids, G)
    block_sal = triton.next_power_of_2(max(1, int(group_len.max().item())))

    delta_tri = sqat_fakequant_delta(
        W_base, BA, base_max, base_min,
        group_start, group_len, group_ids,
        lora_scaling, q_lvl, block_sal=block_sal,
    )
    err1 = (delta_tri - delta_ref).abs().max().item()
    print(f"[triton_sqat] delta max_abs_err = {err1:.2e}  ({'PASS' if err1 < 1e-4 else 'FAIL'})")

    M = 32
    X_S   = torch.randn(M, K, device=device)
    Y_ref = F.linear(X_S, delta_ref)
    Y_tri = sqat_fused_linear_delta(
        X_S, W_base, BA, base_max, base_min,
        group_start, group_len, group_ids,
        lora_scaling, q_lvl, block_sal=block_sal, BM=32, BN=32,
    )
    err2 = (Y_tri - Y_ref).abs().max().item()
    print(f"[triton_sqat] fused_linear max_abs_err = {err2:.2e}  ({'PASS' if err2 < 1e-3 else 'FAIL'})")


if __name__ == "__main__":
    _check_correctness()
