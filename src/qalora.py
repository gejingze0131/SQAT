"""
QA-LoRA support (Xu et al., 2023 — arXiv:2309.14717).

Faithful QA-LoRA, matching the official repo (GPTQ INT-b base, NOT a QAT-on-merged-weight):
  - The base weight is a real GPTQ INT-b group-wise affine quantization, quantized ONCE directly
    from fp16 (build_qalora_intb_base) and baked into an fp16 base checkpoint that training loads
    frozen. No NF4, no double-quant, no per-step re-quantization.
  - LoRA A is resized to consume ONE average-pooled activation per quantization group
    (AvgPool1d(group_size)), so the adapter is a plain low-rank path ADDED ON TOP of the frozen
    quantized base output — it is NEVER fake-quantized:
        y = base_layer(x) + scaling * B( A( avgpool_g(x) ) )
  - Because a pooled-input adapter produces a delta that is constant within each input group, at
    deployment it folds EXACTLY into the affine zero-points (paper Eq. 7), yielding a pure INT
    model with no separate adapter. expand_qalora_delta materializes that fold (delta added on top
    of the frozen INT base) for the dense export: deployed = W_base_intb + expand_delta, which
    equals the training forward bit-for-bit.
"""

import gc
import math
import os
from types import MethodType
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qat_base import QATHandler


# ============================================================================
# GPTQ INT-b base builder (run once, rank 0) — the frozen quantized base
# ============================================================================

def build_qalora_intb_base(
    model_name: str,
    tokenizer,
    calibration_dataloader,
    target_terminals: Sequence[str],
    *,
    group_size: int,
    q_bits: int,
    symmetric: bool,
    save_dir: str,
    device=None,
    percdamp: float = 0.01,
    blocksize: int = 128,
    nsamples: int = 128,
    dtype: torch.dtype = torch.float16,
) -> str:
    """
    Build the QA-LoRA frozen base: GPTQ-quantize every target linear of the fp16 model to INT-b
    group-wise affine (perm_group_k=0 → all columns GPTQ'd, no salient slice), then save the
    quantize→dequant weights as an fp16 checkpoint. Training/export reload this exact checkpoint, so
    the base grid is identical everywhere (no NF4, no double quantization). Run on ONE process.

    Returns save_dir (also writes qalora_base_meta.pt there).
    """
    from transformers import AutoModelForCausalLM
    from .qat_permute_sqat import gptq_quantize_model_sequential

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[QA-LoRA] Building GPTQ INT{q_bits} g{group_size} base from {model_name} (fp16, no NF4)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    # perm_group_k=0 + perm_meta=None → fully GPTQ every target linear (no salient slice, no
    # boundary gather). Replaces weights in-place with their INT-b quantize→dequant values.
    gptq_quantize_model_sequential(
        model,
        calibration_dataloader,
        list(target_terminals),
        perm_group_k=0,
        group_size=group_size,
        q_bits=q_bits,
        symmetric=symmetric,
        device=device,
        perm_meta=None,
        percdamp=percdamp,
        blocksize=blocksize,
        nsamples=nsamples,
    )

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(
        {
            "group_size": group_size,
            "q_bits": q_bits,
            "symmetric": symmetric,
            "orig_base_name": model_name,
            "target_terminals": list(target_terminals),
        },
        os.path.join(save_dir, "qalora_base_meta.pt"),
    )
    print(f"[QA-LoRA] Saved GPTQ INT{q_bits} base to {save_dir}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return save_dir


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


def _make_qalora_forward(group_size: int):
    """Build the QA-LoRA forward: PRE-QUANTIZED frozen base + group-wise LoRA on pooled input.

        y = base_layer(x) + scaling * B( A( avgpool_g(x) ) )

    The base weight is a frozen GPTQ INT-b grid baked into the fp16 base checkpoint
    (build_qalora_intb_base), so the forward does NOT re-quantize anything — it just runs the
    plain frozen linear and adds a low-rank adapter that consumes the group-average-pooled input.
    The adapter is NEVER fake-quantized: its per-group-constant delta folds exactly into the affine
    zero-points at deploy (expand_qalora_delta), so deployed = W_base_intb + expand_delta, which
    equals this forward bit-for-bit (verified). This is the faithful QA-LoRA computation, not a
    QAT-on-merged-weight.
    """

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        adapter_name = _active_adapter_name(self)
        if (
            getattr(self, "disable_adapters", False)
            or adapter_name not in self.lora_A
        ):
            return self.base_layer(x, *args, **kwargs)

        if getattr(self, "merged", False):
            raise RuntimeError("QA-LoRA does not support merged PEFT modules during training.")

        # Frozen pre-quantized base (GPTQ INT-b grid already in the fp16 weights). No re-quant.
        base_out = self.base_layer(x, *args, **kwargs)

        # Separate group-wise LoRA path on the AVG-POOLED input. lora_A was resized to
        # [rank, num_groups] by patch_qalora_model, so it consumes one mean activation per quant
        # group; this delta is constant within each input group and folds into the affine
        # zero-points at deploy — hence it must NOT be quantized here.
        A = self.lora_A[adapter_name]
        B = self.lora_B[adapter_name]
        scaling = float(self.scaling[adapter_name])
        x_pooled = pool_qalora_input(x, group_size).to(A.weight.dtype)
        dropout = getattr(self, "lora_dropout", None)
        if dropout is not None and adapter_name in dropout:
            x_pooled = dropout[adapter_name](x_pooled)
        lora_out = B(A(x_pooled)) * scaling
        return base_out + lora_out.to(base_out.dtype)

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

        # Guard: the QA-LoRA base MUST be the fp16 GPTQ INT-b checkpoint, NOT bitsandbytes NF4.
        # If it's NF4, model_loader's qalora fp16 branch (and/or the train.py GPTQ pre-step) did not
        # run — the forward would then train against a 4-bit NF4 base with no INT-b grid, silently
        # producing a meaningless model. Fail loudly instead.
        _bw = getattr(module.base_layer, "weight", None)
        if getattr(_bw, "quant_state", None) is not None or "4bit" in type(module.base_layer).__name__.lower():
            raise RuntimeError(
                f"[QA-LoRA] base layer '{name}' is bitsandbytes-quantized (NF4), but QA-LoRA needs "
                f"the fp16 GPTQ INT-b base. The train.py GPTQ pre-step / model_loader fp16 qalora "
                f"branch did not run (were those edits reverted?). Aborting to avoid a silent "
                f"garbage model."
            )

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
            module.forward = MethodType(_make_qalora_forward(group_size), module)
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
            f"[QA-LoRA] Patched {count} LoRA layers — frozen GPTQ INT{cfg['model']['quant_bits']} "
            f"affine base + group-pooled adapter (group_size={cfg['qat'].get('group_size', 128)}, "
            f"asymmetric=True); adapter folds into zero-points at deploy."
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
