"""
QEFT — Quantization for Efficient Fine-Tuning (Lee et al., EMNLP 2024 Findings, arXiv:2410.08661),
implemented natively in this repo's framework so its row sits in the same experimental cell as
SALT-Q, QA-LoRA, QWHA and the GPTQ floor.

THE METHOD, IN THREE PARTS (upstream: github.com/xvyaward/qeft)

  1. WEAK COLUMNS, KEPT IN FP16 (from OWQ). Per linear layer, the k input columns whose
     quantization hurts most are left at full precision; everything else is group-wise INT-b.
     Sensitivity is second-order: E_i ≈ ΔW_i,: H ΔW_i,:^T with H = 2 X X^T, so the per-column
     term is λ_j = diag(H)_j (paper Eq. 2-3).

  2. OFFLINE GLOBAL REORDERING (OGR). OWQ's weak columns are scattered, which is why it needs an
     irregular kernel. QEFT observes that the weak indices largely COINCIDE across layers (they
     are driven by activation outliers that propagate through the residual stream) and picks ONE
     GLOBAL index set for the whole network, then folds the corresponding permutation into the
     embedding, both norms per block, q/k/v/gate/up input columns, o_proj/down_proj output rows
     and the lm_head — offline, exactly output-preserving, no runtime gather.

  3. WEAK COLUMN TUNING (WCT). Fine-tuning trains the fp16 weak columns and NOTHING else: the
     quantized 98-99% is frozen, there is no adapter, and the backward only touches a
     [out, k] block, so ∂L/∂W costs k/IC of a dense weight gradient and only the weak columns'
     activations have to be kept (paper §3.3, Fig. 4).

HOW IT DIFFERS FROM SALT-Q, WHICH IS THE POINT OF RUNNING IT HERE

  | | SALT-Q | QEFT |
  |---|---|---|
  | permutation | one P_k PER SEGMENT (+ boundary gathers at runtime) | ONE global P for the whole model, no gather |
  | salient columns | trained as INT-b weights (LSQ fake-quant), deploy on the grid | kept and trained in FP16, never quantized |
  | non-salient | frozen codes + TRAINABLE (s, z) | frozen, zero degrees of freedom |
  | selection | E[x²] per segment, union of per-source top-ratio | Hessian diagonal λ_j, per-layer normalized, summed over the whole model |
  | deployment | pure INT-b | INT-b + k fp16 columns per layer ("b + fp16") |

  Worth stating precisely, because it is easy to overclaim the third row: for a linear layer's
  input, diag(H) = diag(2 X X^T) = 2 Σ_t x_t², i.e. QEFT's λ_j and this repo's E[x²] are the SAME
  statistic up to the constant 2T. The real difference is the AGGREGATION — QEFT normalizes each
  layer's λ by its own mean and sums over every q/k/v/gate/up layer in the network to get one
  global ranking, where SALT-Q takes per-segment unions of per-source top-ratio sets — and the
  fact that QEFT's set is global while SALT-Q's is per segment.

WHAT LIVES WHERE

  qeft_common.py (this file)  selection, the base builder, QEFTLinear, checkpoint I/O
  build_base.py               stage 0 CLI: calibrate → reorder → GPTQ → dense base on disk
  train_commonsense.py        stage 1 CLI: weak-column tuning on this repo's commonsense cell
  export_dense.py             stage 2 CLI: scatter the trained columns back, assert, save
  PROVENANCE.md               what is upstream, what is ours, and why

Everything numerical is shared with the other rows: src/gptq.py does the quantization,
src/permute_common.py the calibration statistics and the permutation folds, src/data.py the
prompt/tokenization/calibration sampling, runs/eval_vllm.sh the scoring.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.permute_common import (  # noqa: E402
    _build_segment_perm,
    _collect_second_moments,
    _resolve_decoder_layers,
    apply_block_internal_permutations_fp32,
    apply_segment_permutation_fp32,
    select_internal_salient_channels,
)

QEFT_META_FILENAME = "qeft_meta.pt"
QEFT_WEAK_FILENAME = "qeft_weak_columns.safetensors"
QEFT_CKPT_META_FILENAME = "qeft_ckpt_meta.json"
QEFT_RUN_META_FILENAME = "qeft_run_meta.json"

WEAK_PARAM_NAME = "weight_weak"

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# Which residual-stream second-moment source each projection reads. o_proj (attention output) and
# down_proj (MLP intermediate) have their own per-layer input space and are handled separately.
RESIDUAL_SOURCE = {"q_proj": "attn", "k_proj": "attn", "v_proj": "attn",
                   "gate_proj": "mlp", "up_proj": "mlp"}


# ============================================================================
# Part 1 — weak column selection
# ============================================================================

def select_global_weak_columns(
    second_moments: Dict[Tuple[int, str], torch.Tensor],
    hidden_size: int,
    num_layers: int,
    k: int,
    targets: Sequence[str] = DEFAULT_TARGETS,
) -> List[int]:
    """QEFT's global weak-column index set (paper §3.2 / Alg. 1; upstream extract_outidx.py).

    For every residual-stream linear in the network, take the Hessian diagonal λ = diag(2 X X^T),
    normalize it by its own mean (so a layer with large activations cannot dominate the vote),
    accumulate, and keep the global top-k:

        s_global += λ^(l,proj) / mean(λ^(l,proj))          for proj in q,k,v,gate,up, all l
        weak      = topk(s_global, k)

    λ_j = 2 Σ_t x_{t,j}², so `second_moments` (E[x²]) is the same vector up to a positive constant
    that the per-layer normalization cancels exactly. q/k/v share one input and gate/up share the
    other, so each source is counted once per projection that is actually being quantized — which
    is upstream's behaviour (it loops over the projections, not over the two inputs).
    """
    mult = {"attn": sum(1 for t in targets if RESIDUAL_SOURCE.get(t) == "attn"),
            "mlp": sum(1 for t in targets if RESIDUAL_SOURCE.get(t) == "mlp")}
    if mult["attn"] == 0 and mult["mlp"] == 0:
        raise ValueError(f"no residual-stream projections in targets={list(targets)}")

    s_global = torch.zeros(hidden_size, dtype=torch.float64)
    n_terms = 0
    for layer in range(num_layers):
        for src in ("attn", "mlp"):
            sm = second_moments.get((layer, src))
            if sm is None or mult[src] == 0:
                continue
            v = sm.detach().double()
            mean = v.mean().clamp(min=1e-30)
            s_global += mult[src] * (v / mean)
            n_terms += mult[src]
    if n_terms == 0:
        raise RuntimeError("no residual second moments were collected")

    weak = sorted(torch.topk(s_global, int(k)).indices.tolist())
    share = float(s_global[torch.tensor(weak)].sum() / s_global.sum())
    print(f"[QEFT] global weak columns: k={k} of {hidden_size} "
          f"({k / hidden_size * 100:.2f}% of the residual stream), "
          f"{n_terms} layer votes, sensitivity share={share * 100:.1f}%, first10={weak[:10]}")
    return weak


def select_local_weak_columns(second_moment: torch.Tensor, k: int) -> List[int]:
    """Per-layer weak columns for a private input space (o_proj, down_proj): plain top-k of λ."""
    k = min(int(k), int(second_moment.numel()))
    return sorted(second_moment.detach().double().topk(k).indices.tolist())


# ============================================================================
# Part 2 — the QEFT layer
# ============================================================================

class QEFTLinear(nn.Module):
    """One linear layer split the way QEFT deploys it: frozen quantized bulk + fp16 weak columns.

        y = x @ W_frozenᵀ + x[:, weak] @ W_weakᵀ

    `W_frozen` holds the dequantized INT-b weight with the weak columns ZEROED (they are carried
    by `weight_weak` instead, so the sum is the effective weight and nothing is double counted).
    `weight_weak` is the ONLY trainable tensor in the module and is kept in fp32 — it is a master
    weight, and one epoch of updates at lr 5e-6 is below bf16's resolution on a ~1e-2 weight.

    The second GEMM is [T, k] × [k, out]: this is the k/IC backward QEFT's §3.3 is about, and with
    the global reordering the weak columns are a leading contiguous slice, so gathering them is a
    view rather than an index_select. o_proj is the exception upstream also carves out (multi-head
    structure forbids cross-head reordering), and there the gather is real.
    """

    def __init__(self, base: nn.Linear, weak_ids: Sequence[int],
                 weak_dtype: torch.dtype = torch.float32):
        super().__init__()
        if base.bias is not None:
            raise NotImplementedError("QEFTLinear assumes no bias (Llama has none)")
        self.in_features = base.in_features
        self.out_features = base.out_features

        ids = torch.as_tensor(sorted(int(i) for i in weak_ids), dtype=torch.long)
        assert ids.numel() > 0 and ids.unique().numel() == ids.numel()
        assert int(ids.max()) < self.in_features
        self.register_buffer("weak_ids", ids, persistent=False)
        self.k = int(ids.numel())
        # OGR's whole point: a contiguous prefix costs a slice, not a gather.
        self.contiguous_prefix = bool(torch.equal(ids, torch.arange(self.k, dtype=torch.long)))

        W = base.weight.data
        self.weight_weak = nn.Parameter(W[:, ids].detach().to(weak_dtype).clone())
        frozen = W.detach().clone()
        frozen[:, ids] = 0
        self.weight = nn.Parameter(frozen, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight)
        xw = x[..., :self.k] if self.contiguous_prefix else x.index_select(-1, self.weak_ids)
        return y + F.linear(xw, self.weight_weak.to(x.dtype))

    @torch.no_grad()
    def effective_weight(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """The dense weight this layer deploys as: frozen bulk with the weak columns written in."""
        dtype = dtype or self.weight.dtype
        W = self.weight.detach().to(dtype).clone()
        W[:, self.weak_ids] = self.weight_weak.detach().to(dtype)
        return W

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, k={self.k}, "
                f"contiguous={self.contiguous_prefix}")


def _split_parent(name: str) -> Tuple[str, str]:
    parent, _, child = name.rpartition(".")
    return parent, child


def qeft_layers(model: nn.Module) -> Dict[str, QEFTLinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, QEFTLinear)}


def install_qeft_layers(model: nn.Module, weak_ids: Dict[str, Sequence[int]],
                        weak_dtype: torch.dtype = torch.float32) -> int:
    """Replace every named linear with a QEFTLinear and freeze everything else."""
    modules = dict(model.named_modules())
    missing = [n for n in weak_ids if n not in modules]
    if missing:
        raise KeyError(f"{len(missing)} modules from the base meta are not in the model, "
                       f"e.g. {missing[:3]}")
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for name, ids in weak_ids.items():
        parent_name, child = _split_parent(name)
        parent = modules[parent_name] if parent_name else model
        setattr(parent, child, QEFTLinear(modules[name], ids, weak_dtype=weak_dtype))
        n += 1
    return n


# ============================================================================
# Part 3 — base builder (stage 0: calibrate → reorder → GPTQ)
# ============================================================================

def _fp16_cost(model_config, targets: Sequence[str], weak_ids: Dict[str, Sequence[int]],
               bits: int) -> Tuple[float, float]:
    """fp16 share of the TARGET weights and the resulting effective bit width.

    The QEFT row is not a b-bit row and the tables must not read it as one: k fp16 columns per
    layer is an honest surcharge that has to be quoted next to the accuracy.
    """
    total = fp16 = 0
    h, i = model_config.hidden_size, model_config.intermediate_size
    shapes = {"q_proj": (h, h), "k_proj": (h, h), "v_proj": (h, h), "o_proj": (h, h),
              "gate_proj": (i, h), "up_proj": (i, h), "down_proj": (h, i)}
    n_layers = model_config.num_hidden_layers
    kv = getattr(model_config, "num_key_value_heads", model_config.num_attention_heads)
    if kv != model_config.num_attention_heads:                       # GQA: k/v are narrower
        head = h // model_config.num_attention_heads
        shapes["k_proj"] = (kv * head, h)
        shapes["v_proj"] = (kv * head, h)
    for t in targets:
        out_f, in_f = shapes[t]
        total += out_f * in_f * n_layers
    for name, ids in weak_ids.items():
        term = name.split(".")[-1]
        fp16 += shapes[term][0] * len(ids)
    share = fp16 / max(total, 1)
    return share, share * 16.0 + (1.0 - share) * bits


@torch.no_grad()
def build_qeft_base(
    model_name: str,
    tokenizer,
    calibration_dataloader,
    save_dir: str,
    *,
    k: int,
    group_size: int,
    q_bits: int,
    symmetric: bool = False,
    targets: Sequence[str] = DEFAULT_TARGETS,
    oproj_weak: bool = True,
    percdamp: float = 0.01,
    blocksize: int = 128,
    nsamples: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> dict:
    """Stage 0, ONE process, one GPU. Produces the model QEFT fine-tunes.

      1. calibrate λ = diag(2 X X^T) for every linear's input (src/permute_common)
      2. pick the ONE global weak-column set and fold its permutation into the whole network
         (embedding, norms, q/k/v/gate/up columns, o_proj/down_proj rows, lm_head) — offline and
         output-preserving, with a single segment there is no runtime gather at all
      3. per-layer weak columns for the two private input spaces: down_proj gets an offline
         internal permutation (gate/up output rows follow), o_proj keeps its irregular indices
      4. GPTQ every target linear at INT-b group-wise asymmetric with the weak columns EXCLUDED
         and left at full precision (src/gptq.gptq_quantize_layer's keep_salient_fp16 path)
      5. save the dense checkpoint + qeft_meta.pt

    The saved checkpoint is a plain Llama: vLLM can serve it, and the fine-tuning script only has
    to know which columns are the weak ones.
    """
    import gc
    from transformers import AutoModelForCausalLM

    from src.gptq import gptq_quantize_model_sequential

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = list(targets)
    if k % group_size:
        raise ValueError(
            f"k={k} must be a multiple of group_size={group_size}: the weak columns occupy whole "
            f"quantization groups, and src/gptq.py asserts it (a group cannot straddle the "
            f"fp16/INT boundary — the scale/zp layout has no way to represent that).")

    print(f"[QEFT] loading {model_name} in {dtype} (no bnb, no adapter)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    model.to(device).eval()

    d_model = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    if k > d_model:
        raise ValueError(f"k={k} exceeds hidden_size={d_model}")

    # ---- 1) calibration ------------------------------------------------------------------
    second_moments = _collect_second_moments(
        model, calibration_dataloader, num_layers, device, collect_internal=True)

    # ---- 2) offline global reordering ------------------------------------------------------
    global_weak = select_global_weak_columns(second_moments, d_model, num_layers, k, targets)
    residual_perm = _build_segment_perm(global_weak, d_model)
    boundary_perms = apply_segment_permutation_fp32(model, {0: residual_perm}, [num_layers])
    assert not boundary_perms, "OGR is a single segment — there must be no runtime gather"

    # ---- 3) the two private input spaces ---------------------------------------------------
    down_perms: Dict[int, List[int]] = {}
    if "down_proj" in targets:
        internal = select_internal_salient_channels(
            second_moments, num_layers, group_k=k, down_layer_group_ks=[k] * num_layers)
        apply_block_internal_permutations_fp32(model, internal)
        down_perms = {l: list(v) for (l, _t), v in internal.items()}

    oproj_ids: Dict[int, List[int]] = {}
    if oproj_weak and "o_proj" in targets:
        # o_proj cannot be reordered: its input is the concatenated attention heads, and mixing
        # channels across heads is not an equivalence transform. Upstream reorders it ONLINE in
        # its kernel; keeping the irregular indices in place is the same arithmetic without the
        # gather, and src/gptq.py's `salient_ids` quantizes around them.
        for l in range(num_layers):
            sm = second_moments.get((l, "o_proj"))
            if sm is not None:
                oproj_ids[l] = select_local_weak_columns(sm, k)

    # ---- weak column index per module, in the DEPLOYED basis --------------------------------
    layers = _resolve_decoder_layers(model)
    name_of = {m: n for n, m in model.named_modules()}
    weak_ids: Dict[str, List[int]] = {}
    salient_ids: Dict[str, List[int]] = {}
    for l in range(num_layers):
        for sub in layers[l].modules():
            if not isinstance(sub, nn.Linear):
                continue
            nm = name_of[sub]
            term = nm.split(".")[-1]
            if term not in targets:
                continue
            if term == "o_proj":
                if l in oproj_ids:
                    weak_ids[nm] = list(oproj_ids[l])
                    salient_ids[nm] = list(oproj_ids[l])       # irregular → GPTQ needs the set
            else:
                weak_ids[nm] = list(range(min(k, sub.in_features)))

    share, eff_bits = _fp16_cost(model.config, targets, weak_ids, q_bits)
    print(f"[QEFT] fp16 share of the target weights = {share * 100:.2f}% "
          f"→ effective {eff_bits:.2f} bits (the row is '{q_bits} + fp16', not {q_bits}-bit)")

    # perm_meta shape src/gptq.py understands: layer_group_ks drives the leading fp16 slice.
    perm_meta = {
        "boundary_perms": [],
        "boundary_layer_indices": [],
        "segment_perms": {0: list(residual_perm)},
        "block_internal_perms": {f"{l}_down_proj": v for l, v in down_perms.items()},
        "group_k": int(k),
        "fixed_group_k": int(k),
        "segment_group_ks": [int(k)],
        "layer_group_ks": [int(k)] * num_layers,
        "down_layer_group_ks": [int(k)] * num_layers,
        "boundary_sizes": [num_layers],
        "group_size": int(group_size),
        "d_model": d_model,
    }

    # ---- 4) GPTQ, weak columns excluded and kept at full precision --------------------------
    # Kept in the model's own dtype, not fp32: this is an exact copy either way (the weights ARE
    # bf16 here) and fp32 would double a 13 GB host-side snapshot on the prep node.
    ref = {n: p.detach().cpu().clone()
           for n, p in model.named_parameters()
           if n.endswith(".weight") and n.split(".")[-2] in targets}

    gptq_quantize_model_sequential(
        model, calibration_dataloader, targets,
        perm_group_k=int(k),
        group_size=int(group_size),
        q_bits=int(q_bits),
        symmetric=bool(symmetric),
        device=device,
        perm_meta=perm_meta,
        percdamp=percdamp,
        blocksize=blocksize,
        nsamples=nsamples,
        awq_scales=None,
        keep_salient_fp16=True,          # the weak columns ARE the method
        salient_ids=salient_ids or None,  # o_proj's irregular set
    )

    # Two errors, because one must be exactly 0 and the other must not be: the weak columns are
    # supposed to come back untouched, and the rest is supposed to carry the INT-b error.
    weak_drift, bulk_err = [], []
    params = dict(model.named_parameters())
    for name, r0 in ref.items():
        mod_name = name[: -len(".weight")]
        r = r0.float()
        q = params[name].detach().float().cpu()
        ids = torch.as_tensor(weak_ids.get(mod_name, []), dtype=torch.long)
        if ids.numel():
            weak_drift.append((r[:, ids] - q[:, ids]).abs().max().item())
        mask = torch.ones(r.shape[1], dtype=torch.bool)
        mask[ids] = False
        bulk_err.append((r[:, mask] - q[:, mask]).norm().item()
                        / max(r[:, mask].norm().item(), 1e-12))
    bulk_err.sort()
    print(f"[QEFT] weak-column max drift = {max(weak_drift):.3e} (must be 0) | "
          f"quantized-bulk relative error p50 = {bulk_err[len(bulk_err) // 2]:.4f}")
    if max(weak_drift) > 0:
        raise RuntimeError("the fp16 weak columns were modified by the quantizer")
    if bulk_err[len(bulk_err) // 2] < 1e-3:
        raise RuntimeError("the quantized bulk has ~0 relative error — nothing was quantized")

    # ---- 5) save ---------------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    meta = {
        "model_name": model_name,
        "q_bits": int(q_bits),
        "group_size": int(group_size),
        "symmetric": bool(symmetric),
        "k": int(k),
        "targets": targets,
        "oproj_weak": bool(oproj_weak and bool(oproj_ids)),
        "global_weak_columns": list(global_weak),
        "residual_perm": list(residual_perm),
        "down_perms": down_perms,
        "oproj_weak_ids": oproj_ids,
        "weak_ids": weak_ids,
        "d_model": d_model,
        "num_layers": num_layers,
        "dtype": str(dtype).replace("torch.", ""),
        "fp16_share": share,
        "effective_bits": eff_bits,
        "gptq": {"nsamples": nsamples, "percdamp": percdamp, "blocksize": blocksize},
        "base_dir": os.path.abspath(save_dir),
    }
    torch.save(meta, os.path.join(save_dir, QEFT_META_FILENAME))
    print(f"[QEFT] base saved → {save_dir}")

    del model, ref
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


# ============================================================================
# Part 4 — model build + checkpoint I/O
# ============================================================================

def load_qeft_meta(base_dir: str) -> dict:
    path = os.path.join(base_dir, QEFT_META_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no {QEFT_META_FILENAME} in {base_dir} — build the base first "
                                f"(baseline/QEFT/sqat/build_base.py)")
    return torch.load(path, map_location="cpu", weights_only=False)


def build_qeft_model(base_dir: str, *, dtype: torch.dtype = torch.bfloat16,
                     device=None, weak_dtype: torch.dtype = torch.float32):
    """Load the QEFT base and turn its target linears into QEFTLinear. Returns (model, meta)."""
    from transformers import AutoModelForCausalLM

    meta = load_qeft_meta(base_dir)
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    n = install_qeft_layers(model, meta["weak_ids"], weak_dtype=weak_dtype)
    if device is not None:
        model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[QEFT] {n} QEFTLinear layers | trainable {trainable / 1e6:.1f}M / {total / 1e6:.1f}M "
          f"({trainable / max(total, 1) * 100:.2f}%) | k={meta['k']} INT{meta['q_bits']} "
          f"g{meta['group_size']}")
    return model, meta


def qeft_trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {f"{n}.{WEAK_PARAM_NAME}": m.weight_weak.detach().cpu()
            for n, m in qeft_layers(model).items()}


def save_qeft_trainable(model: nn.Module, output_dir: str, base_dir: Optional[str] = None,
                        extra: Optional[dict] = None) -> str:
    """Write ONLY the trained weak columns (~0.7 GB at k=128) plus a pointer to their base.

    The frozen bulk is several GB, never changes, and already sits in the base directory; a
    checkpoint that copied it would make every save cost more than the epoch.
    """
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)
    sd = qeft_trainable_state_dict(model)
    save_file({k: v.contiguous() for k, v in sd.items()},
              os.path.join(output_dir, QEFT_WEAK_FILENAME))
    meta = {"base_dir": os.path.abspath(base_dir) if base_dir else None,
            "num_tensors": len(sd),
            "num_params": int(sum(v.numel() for v in sd.values()))}
    if extra:
        meta.update(extra)
    with open(os.path.join(output_dir, QEFT_CKPT_META_FILENAME), "w") as f:
        json.dump(meta, f, indent=2)
    return output_dir


def load_qeft_trainable(model: nn.Module, checkpoint_dir: str) -> int:
    from safetensors.torch import load_file

    path = os.path.join(checkpoint_dir, QEFT_WEAK_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no {QEFT_WEAK_FILENAME} in {checkpoint_dir}")
    sd = load_file(path)
    layers = qeft_layers(model)
    n = 0
    for name, mod in layers.items():
        key = f"{name}.{WEAK_PARAM_NAME}"
        if key not in sd:
            raise KeyError(f"{key} missing from {path}")
        t = sd[key]
        if tuple(t.shape) != tuple(mod.weight_weak.shape):
            raise ValueError(f"{key}: checkpoint {tuple(t.shape)} != model "
                             f"{tuple(mod.weight_weak.shape)} — wrong base?")
        with torch.no_grad():
            mod.weight_weak.copy_(t.to(mod.weight_weak.dtype).to(mod.weight_weak.device))
        n += 1
    if n != len(sd):
        raise ValueError(f"loaded {n} tensors but the checkpoint has {len(sd)}")
    return n


def qeft_base_dir_from_checkpoint(checkpoint_dir: str) -> Optional[str]:
    path = os.path.join(checkpoint_dir, QEFT_CKPT_META_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("base_dir")


# ============================================================================
# Part 5 — data (this repo's cell, imported rather than copied)
# ============================================================================

def load_sqat_data_module(cfg: dict, tokenizer):
    """Tokenized train set + collator from src/data.py — the same prompt, EOS convention and
    loss_span every other row in the table was trained with."""
    from src import data as sqat_data

    train_dataset, eval_dataset = sqat_data.load_dataset_for_training(cfg, tokenizer)
    return train_dataset, eval_dataset, sqat_data.build_data_collator(tokenizer)


def build_calibration_loader(cfg: dict, tokenizer, batch_size: Optional[int] = None):
    """The balanced in-domain calibration set every `bcal` row is defined by (src/data.py).

    Both offline consumers read it: the λ statistics that choose the weak columns and the GPTQ
    Hessians that quantize everything else. qat.gptq.nsamples must equal
    qat.sqat.calibration_samples — gptq_quantize_model_sequential stops after the first nsamples
    records, so a smaller value would silently quantize on a subset of the calibration set.
    """
    from torch.utils.data import DataLoader

    from src.data import build_data_collator, load_calibration_data

    ds = load_calibration_data(cfg, tokenizer)
    bs = int(batch_size or (cfg["qat"].get("gptq", {}) or {}).get("batch_size", 8))
    return DataLoader(ds, batch_size=bs, collate_fn=build_data_collator(tokenizer), shuffle=False)
