"""
gptq.py — GPTQ / OBS quantization, shared by every method that needs a strong PTQ initialization.

Consumers:
  * `qat_permute_sqat` export — the ~98% NON-salient columns get OBS error compensation instead
    of RTN, while the salient slice stays pinned to the canonical (training-consistent) grid.
  * `qalora.build_qalora_intb_base` — perm_group_k=0, i.e. the whole weight is GPTQ'd, producing
    the frozen INT-b base QA-LoRA trains against.
  * `qat_saltq.build_saltq_base` — the same call produces SALT-Q's FROZEN integer codes and the
    initial (s, z) for the non-salient segment.

Two invariants that cost real debugging time:

  1. The salient slice's quantization error is NOT propagated into the non-salient columns.
     Training saw (fakequant'd salient + un-requantized fp16 non-salient), so the deployed
     non-salient columns must approximate that SAME fp16 weight — not absorb the salient error.
     GPTQ therefore runs as an INDEPENDENT OBS problem on H[group_k:, group_k:].

  2. STATIC groups: every group's scale/zp is precomputed from the ORIGINAL weights and held
     FIXED through the sweep. Recomputing a grid mid-sweep from partially-updated weights breaks
     OBS's fixed-grid assumption and makes the compensation INCREASE output error (GPTQ worse
     than RTN), which gets worse as groups grow and bits shrink.

The Hessian H = X^T X must be collected in the SAME (permuted) basis as the weight columns, with
the boundary gathers registered — `gptq_quantize_model_sequential` does this for you.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .permute_common import (
    _resolve_decoder_layers,
    awq_s_for_module,
    group_k_for_module_name,
    register_boundary_gathers_from_meta,
)
from .quant_primitives import (
    _asym_q_max,
    _asym_qparams,
    _sym_q_max,
    _sym_scale,
    group_dequantize,
    group_quantize,
    lsq_scale_for_module,
)


# ============================================================================
# Part 8b — GPTQ (Optimal Brain Quantization) for the NON-salient columns.
#
# Improvement over plain RTN export: the ~97% non-salient columns are quantized with OBS error
# compensation instead of round-to-nearest. The salient slice [0:layer_group_k] is the QAT-protected
# part and MUST keep the EXACT canonical group_quantize grid the LoRA was trained against (else
# the QAT benefit does not transfer — the earlier min/max-vs-pos/neg regression). So GPTQ here:
#   * fixes columns [0:layer_group_k] to the canonical RTN grid (training-consistent), and
#   * GPTQ-quantizes columns [layer_group_k:] with OBS compensation (same group_size and asym/sym as QAT).
# o_proj carries no salient slice (group_k=0) → it is fully GPTQ-quantized.
#
# The Hessian H = X^T X must be in the SAME (permuted) basis as the weight columns — collect it on
# the permuted base with the boundary gathers registered (gptq_quantize_model_sequential does this).
# ============================================================================

def _gptq_cholesky_inv_upper(H: torch.Tensor, percdamp: float, max_tries: int = 5) -> torch.Tensor:
    """
    Return the upper-triangular Cholesky factor U of H^{-1} (H^{-1} = U^T U), the form GPTQ's
    sequential update consumes. Dead (zero-activation) columns are made invertible; damping is
    escalated until H+damp is positive-definite.
    """
    cols = H.shape[0]
    H = H.clone()
    diagH = torch.diagonal(H)
    dead = diagH == 0.0
    if dead.any():
        H[dead, dead] = 1.0
        diagH = torch.diagonal(H)
    live = ~dead
    mean_diag = diagH[live].mean() if live.any() else H.new_tensor(1.0)
    idx = torch.arange(cols, device=H.device)
    base = percdamp * mean_diag
    for t in range(max_tries):
        Hd = H.clone()
        Hd[idx, idx] += base * (1.0 + t)            # escalate damping if not PD
        try:
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.cholesky_inverse(L)
            return torch.linalg.cholesky(Hinv, upper=True)
        except RuntimeError:
            continue
    raise RuntimeError("GPTQ: Hessian Cholesky failed even after damping escalation.")


@torch.no_grad()
def gptq_quantize_layer(
    W: torch.Tensor,            # [out, in] dense weight (permuted basis)
    H: torch.Tensor,            # [in, in]  input Hessian (X^T X) in the SAME basis
    group_k: int,               # leading salient columns held at the canonical grid (0 = none)
    group_size: int,
    q_bits: int,
    symmetric: bool,
    percdamp: float = 0.01,
    blocksize: int = 128,
    eps: float = 1e-8,
    awq_s: Optional[torch.Tensor] = None,   # [group_k] AWQ scale for the salient slice, or None
    keep_salient_fp16: bool = False,        # ablation: leave the salient slice as fp16 (no quant)
    fixed_scale=None,                       # LSQ learned scale[, zp] for the salient slice, or None
    obs_salient: bool = False,              # SALT-Q salient_init=gptq: ONE OBS problem over all columns
):
    """
    Quantize the salient slice [0:group_k] to the canonical SQAT grid (training-consistent) and
    GPTQ-quantize ONLY the non-salient block [group_k:]. Returns (W_int [out,in], scale [out,ng],
    zp [out,ng]) in the EXACT layout of group_quantize, so the existing group_dequantize path
    reconstructs the deployed weight unchanged.

    IMPORTANT — the salient slice's quantization error is NOT propagated into the non-salient
    columns. The QAT/LoRA was trained to tolerate the salient slice's quant error (training saw
    the fakequant'd salient + the un-requantized fp16 non-salient), so the deployed non-salient
    must approximate that SAME fp16 weight W_n — not "absorb" the salient error (doing so shifts
    the deployed output away from the training-time output and degrades accuracy). GPTQ therefore
    runs as an INDEPENDENT OBS problem on the non-salient block with the non-salient sub-Hessian
    H[group_k:, group_k:], minimizing ||(W_q_n - W_n) X_n|| — strictly an improvement over RTN.

    AWQ-style scaling: if `awq_s` is given, the salient slice is quantized in the amplified space
    (W_S * S). The STORED ints/scale/zp stay in the amplified space — the caller bakes the `/S`
    back into the salient columns of the dequantized dense weight (see _unscale_salient_cols in
    export.py), matching the training fakequant W_fq/S exactly. AWQ only touches the salient slice.
    """
    dev = W.device
    out_f, in_f = W.shape
    assert in_f % group_size == 0, \
        f"GPTQ requires in_features ({in_f}) divisible by group_size ({group_size})"
    assert group_k % group_size == 0, \
        f"group_k ({group_k}) must be a multiple of group_size ({group_size})"
    if obs_salient and group_k > 0:
        # SALT-Q salient_init=gptq (src/qat_saltq.build_saltq_base): the salient slice is NOT pinned
        # to the RTN grid. The whole matrix is one OBS problem — the salient columns come first and
        # their rounding error is compensated by the columns after them, exactly as in a plain GPTQ
        # export — and the caller keeps the salient part of (W_int, scale, zp) as the trainable LSQ
        # start. The layout of the returned tensors is unchanged.
        assert awq_s is None and fixed_scale is None and not keep_salient_fp16, \
            "obs_salient composes with none of awq_s / fixed_scale / keep_salient_fp16"
        group_k = 0
    ng = in_f // group_size
    q_max = _sym_q_max(q_bits) if symmetric else _asym_q_max(q_bits)

    # block size aligned to group_size so a quant group never straddles a block boundary
    if blocksize < group_size:
        blocksize = group_size
    blocksize = (blocksize // group_size) * group_size

    W = W.clone().float()
    H = H.float().to(dev)

    W_int = torch.zeros(out_f, in_f, device=dev)
    scale = torch.zeros(out_f, ng, device=dev)
    zp    = torch.zeros(out_f, ng, device=dev)

    # ---- 1) salient slice [0:group_k]: fixed canonical grid (amplified if AWQ). NO propagation. ----
    # ABLATION (keep_salient_fp16): the salient slice is NOT quantized at all — it is deployed at
    # fp16 (the methodology upper bound: "what if the QAT-protected slice were full precision?").
    # We leave W_int/scale/zp at zero for those leading columns; the caller restores the original
    # fp16 weight into the dequantized dense weight. The non-salient GPTQ block below is identical
    # either way (it already targets the fp16 W_n with the independent non-salient sub-Hessian).
    n_sal_g = group_k // group_size
    if group_k > 0 and not keep_salient_fp16:
        W_sal = W[:, :group_k]
        # LSQ: fixed_scale carries the learned scale[, zp]; group_quantize then uses the LSQ grid
        # for the salient slice (identical to the training fakequant). Move to W's device.
        fs = fixed_scale
        if fs is not None:
            fs = fs.to(dev) if torch.is_tensor(fs) else (fs[0].to(dev), fs[1].to(dev))
        if awq_s is not None:
            s = awq_s.to(torch.float32).view(1, -1).to(dev)
            wi_s, sc_s, zp_s = group_quantize(W_sal * s, group_size, q_bits, symmetric, eps,
                                              fixed_scale=fs)
        else:
            wi_s, sc_s, zp_s = group_quantize(W_sal, group_size, q_bits, symmetric, eps,
                                              fixed_scale=fs)
        W_int[:, :group_k] = wi_s.to(dev)
        scale[:, :n_sal_g]  = sc_s.to(dev)
        zp[:, :n_sal_g]     = zp_s.to(dev)

    if group_k >= in_f:                          # nothing non-salient to GPTQ (shouldn't happen)
        return W_int, scale, zp

    # ---- 2) GPTQ on the NON-salient block [group_k:] only (target the fp16 W_n). ----
    Wn = W[:, group_k:]                          # [out, in_n]  (never touched by the salient slice)
    Hn = H[group_k:, group_k:]
    in_n = in_f - group_k

    # STATIC groups: precompute every group's scale/zp from the ORIGINAL weights and keep them
    # FIXED during the sweep. GPTQ updates the not-yet-quantized columns (error compensation), so
    # a grid recomputed mid-sweep from the partially-updated weights is "stale" for the rest of the
    # group — that breaks OBS's fixed-grid assumption and makes the compensation INCREASE the output
    # error (GPTQ worse than RTN), worsening as the group grows / bits shrink. A fixed grid restores
    # OBS optimality (GPTQ ≤ RTN). (Standard GPTQ "static_groups".)
    ng_n = in_n // group_size
    for gi in range(ng_n):
        Wg = Wn[:, gi * group_size:(gi + 1) * group_size]
        if symmetric:
            s = _sym_scale(Wg, q_max, eps); z = torch.zeros_like(s)
        else:
            s, z = _asym_qparams(Wg, q_max, eps)
        scale[:, n_sal_g + gi] = s.squeeze(-1)
        zp[:, n_sal_g + gi]    = z.squeeze(-1)

    Hinv = _gptq_cholesky_inv_upper(Hn, percdamp)

    for i1 in range(0, in_n, blocksize):
        i2 = min(i1 + blocksize, in_n)
        W1    = Wn[:, i1:i2].clone()
        Err1  = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for j in range(i2 - i1):
            col = i1 + j                         # local column within the non-salient block
            g   = n_sal_g + col // group_size    # global group index
            w   = W1[:, j]
            d   = Hinv1[j, j]

            s = scale[:, g].unsqueeze(-1)        # FIXED grid (precomputed above)
            z = zp[:, g].unsqueeze(-1)
            if symmetric:
                qi = torch.round(torch.clamp(w.unsqueeze(-1) / s, -q_max, q_max))
                q  = (qi * s).squeeze(-1)
            else:
                qi = torch.round(torch.clamp(w.unsqueeze(-1) / s + z, 0, q_max))
                q  = ((qi - z) * s).squeeze(-1)
            W_int[:, group_k + col] = qi.squeeze(-1)

            err = (w - q) / d
            W1[:, j:] -= err.unsqueeze(-1) * Hinv1[j, j:].unsqueeze(0)
            Err1[:, j] = err

        Wn[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return W_int, scale, zp


class _GPTQCatcherStop(Exception):
    """Raised by the layer-0 catcher to stop the forward after capturing the first input."""


@torch.no_grad()
def _masked_xtx(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """X^T X over the REAL token positions of a [B, T, d] activation. The collator right-pads
    every batch; before 2026-08-28 the padded positions (the pad/EOS embedding run through the
    stack) went into every Hessian too. With ~74-token records and batch 2 that was a few percent
    of the rows; at the larger calibration batches the budget fix uses it would not be."""
    d = x.shape[-1]
    x = x.reshape(-1, d)
    if mask is not None:
        m = mask.reshape(-1).bool()
        if m.numel() == x.shape[0]:
            x = x[m]
    x = x.float()
    return x.t() @ x


def _front_permutation(ids: Sequence[int], n_cols: int, device) -> torch.Tensor:
    """Column order [salient..., the rest in their original order] for an arbitrary salient set.

    The ids are sorted first, so the permutation is a pure function of the SET rather than of the
    order the selector happened to emit, and a salient set that IS a leading contiguous block
    reproduces the identity permutation exactly.
    """
    ids_t = torch.as_tensor(list(ids), dtype=torch.long, device=device)
    assert ids_t.numel() > 0, "empty salient id list"
    assert ids_t.unique().numel() == ids_t.numel(), "duplicate salient column ids"
    assert int(ids_t.max()) < n_cols and int(ids_t.min()) >= 0, "salient column id out of range"
    ids_t = torch.sort(ids_t).values
    mask = torch.ones(n_cols, dtype=torch.bool, device=device)
    mask[ids_t] = False
    return torch.cat([ids_t, torch.arange(n_cols, device=device)[mask]])


def gptq_quantize_model_sequential(
    model: nn.Module,
    calibration_dataloader: DataLoader,
    target_terminals: Sequence[str],
    perm_group_k: int,
    group_size: int,
    q_bits: int,
    symmetric: bool,
    device: torch.device,
    perm_meta=None,
    percdamp: float = 0.01,
    blocksize: int = 128,
    nsamples: int = 128,
    awq_scales: Optional[dict] = None,
    keep_salient_fp16: bool = False,
    lsq_scales: Optional[dict] = None,
    obs_salient: bool = False,
    salient_ids: Optional[Dict[str, Sequence[int]]] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    In-place sequential GPTQ on a dense (permuted) fp16 model. For every nn.Linear whose terminal
    name is in `target_terminals`:
       * q/k/v/gate/up/down_proj → columns [0:layer_group_k] fixed to the canonical SQAT grid, the
         rest GPTQ-quantized. layer_group_k is read from perm_meta when present, otherwise
         perm_group_k is used for backward compatibility;
       * o_proj                  → no salient slice (group_k=0) → fully GPTQ.
    Each decoder layer is quantized using the ALREADY-quantized previous layers' outputs (true
    cross-layer sequential GPTQ): weights are replaced in place with their quantize→dequant values,
    and the per-layer (W_int, scale, zp) are returned (on CPU) in the group_quantize layout.

    The model MUST be the permuted base; boundary gathers from `perm_meta` are registered for the
    duration so the captured activations are in the deployment basis (and per-layer inputs are
    re-ordered at segment boundaries exactly as at inference).

    `salient_ids` (opt-in, used by the QEFT baseline) names an ARBITRARY, not necessarily
    contiguous, salient column set per module — {full_module_name: [col, ...]}. Those columns are
    moved to the front of an INTERNAL column permutation together with the matching rows/columns
    of the Hessian, quantized exactly as a leading slice would be, and the dequantized weight is
    permuted back before it is written, so the model itself is never reordered. A module named
    here overrides its perm_meta / o_proj group_k. Because the quantization groups then follow the
    PERMUTED column order, the (W_int, scale, zp) returned for such a module are in that permuted
    order too; the dense weight left in the model is in the model's own order either way. Modules
    absent from the dict (and the default None) behave exactly as before.
    """
    target_terminals = set(target_terminals)
    name_of = {m: n for n, m in model.named_modules()}
    layers = _resolve_decoder_layers(model)
    num_layers = len(layers)

    prev_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    model.eval()

    gather_hooks = register_boundary_gathers_from_meta(model, perm_meta) if perm_meta else []

    # ---- capture layer-0 input + per-batch kwargs (attention_mask / position_embeddings / ...) ----
    inps: List[torch.Tensor] = []
    kwargs_list: List[dict] = []
    orig_layer0 = layers[0]

    class _Catcher(nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, hidden_states, **kw):
            inps.append(hidden_states.detach().to("cpu"))
            kwargs_list.append(
                {k: (v.detach().to("cpu") if torch.is_tensor(v) else v) for k, v in kw.items()}
            )
            raise _GPTQCatcherStop()

    layers[0] = _Catcher(orig_layer0)
    seen = 0
    masks: List[Optional[torch.Tensor]] = []          # 2-D [B, T] padding masks, one per batch
    n_tokens = 0
    for batch in calibration_dataloader:
        if seen >= nsamples:
            break
        ids = batch["input_ids"].to(device)
        am  = batch.get("attention_mask")
        am  = am.to(device) if am is not None else None
        try:
            model(input_ids=ids, attention_mask=am)
        except _GPTQCatcherStop:
            pass
        masks.append(am.detach().to("cpu") if am is not None else None)
        n_tokens += int(am.sum().item()) if am is not None else ids.numel()
        seen += ids.shape[0]
    layers[0] = orig_layer0
    print(f"[GPTQ] Captured {len(inps)} calibration batches ({seen} sequences, {n_tokens} real "
          f"tokens; padded positions are excluded from every Hessian).")

    def _kw_to_dev(kw):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kw.items()}

    quantized_layers: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for L in tqdm(range(num_layers), desc="[GPTQ] Sequential quantize"):
        layer = layers[L]
        subs = {}                                   # full_name -> (module, group_k)
        for sub in layer.modules():
            if isinstance(sub, nn.Linear) and name_of[sub].split(".")[-1] in target_terminals:
                term = name_of[sub].split(".")[-1]
                nm = name_of[sub]
                gk = 0 if term == "o_proj" else group_k_for_module_name(
                    nm, perm_meta=perm_meta, default_group_k=perm_group_k,
                )
                subs[nm] = (sub, gk)
        if not subs:
            continue

        # 1) accumulate input Hessians for this layer's sublayers (fp16 weights)
        Hs: Dict[str, torch.Tensor] = {}
        handles = []

        cur = {"mask": None}                       # set per batch below; read by the hooks

        def _mk(nm):
            def _h(mod, inp, out):
                xtx = _masked_xtx(inp[0].detach(), cur["mask"])
                Hs[nm] = xtx if nm not in Hs else Hs[nm].add_(xtx)
            return _h

        for nm, (mod, _) in subs.items():
            handles.append(mod.register_forward_hook(_mk(nm)))
        for i in range(len(inps)):
            cur["mask"] = masks[i].to(device) if masks[i] is not None else None
            layer(inps[i].to(device), **_kw_to_dev(kwargs_list[i]))
        cur["mask"] = None
        for h in handles:
            h.remove()

        # 2) GPTQ each sublayer; replace its weight with quantize->dequant
        for nm, (mod, gk) in subs.items():
            W = mod.weight.data.float()
            H = Hs[nm]
            # Arbitrary (non-contiguous) salient set: quantize in a column order that puts it
            # first, then undo. Nothing else in this loop has to know about it.
            col_perm = None
            if salient_ids is not None and nm in salient_ids:
                col_perm = _front_permutation(salient_ids[nm], W.shape[1], W.device)
                gk = len(salient_ids[nm])
                W = W[:, col_perm]
                H = H[col_perm][:, col_perm]
            awq_s = awq_s_for_module(awq_scales, nm, gk) if gk > 0 else None
            # LSQ: salient slice uses the learned scale[, zp] (o_proj gk=0 → none).
            fixed_scale = (lsq_scale_for_module(lsq_scales, nm, symmetric)
                           if (lsq_scales and gk > 0) else None)
            W_int, sc, zp = gptq_quantize_layer(
                W, H, gk, group_size, q_bits, symmetric,
                percdamp=percdamp, blocksize=blocksize, awq_s=awq_s,
                keep_salient_fp16=keep_salient_fp16,
                fixed_scale=fixed_scale,
                obs_salient=obs_salient,
            )
            W_deq = group_dequantize(W_int, sc, zp, group_size, W.shape[1], symmetric)
            if keep_salient_fp16 and gk > 0:
                # Ablation: restore the un-quantized fp16 salient slice (group_dequantize left it 0).
                # The cross-layer propagation below then sees the salient slice at full precision.
                W_deq[:, :gk] = W[:, :gk]
            if col_perm is not None:
                W_deq = W_deq[:, torch.argsort(col_perm)]
            if awq_s is not None:
                # bake 1/S back into the salient columns so the in-place (and exported) dense
                # weight is the deployed value W_fq/S — matching the training fakequant.
                W_deq[:, :gk] = W_deq[:, :gk] / awq_s.view(1, -1).to(W_deq.device)
            mod.weight.data.copy_(W_deq.to(mod.weight.dtype))
            quantized_layers[nm] = (W_int.cpu(), sc.cpu(), zp.cpu())
            Hs[nm] = None

        # 3) recompute inputs for the next layer using the QUANTIZED layer
        if L < num_layers - 1:
            for i in range(len(inps)):
                out = layer(inps[i].to(device), **_kw_to_dev(kwargs_list[i]))
                out = out[0] if isinstance(out, tuple) else out
                inps[i] = out.detach().to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for h in gather_hooks:
        h.remove()
    if prev_use_cache is not None:
        model.config.use_cache = prev_use_cache

    return quantized_layers
