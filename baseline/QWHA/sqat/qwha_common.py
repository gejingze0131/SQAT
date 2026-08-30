"""
The seam between the official QWHA code (vantaa89/QWHA @ fc8d288) and this repo.

Everything method-specific -- the Walsh-Hadamard adapter, the AdaAlloc initialization, the peft
fork -- is upstream code, imported and called unchanged. This module only supplies what a
cluster reproduction needs and upstream hard-codes:

  * an EXPLICIT path for the quantized base and an EXPLICIT device. `utils.get_quantized_peft_model`
    keys the GPTQ cache by model id (so a second group size overwrites the first) and ends with
    `.cuda()` / `device_map="auto"`, neither of which survives DDP, where every rank needs its own
    full copy of the base on its own GPU.
  * an index-buffer fix. Upstream creates the adapter with the config's flat `n_frequency`, then
    rewrites `qwha_spectrum` / `qwha_indices` per layer to the LoRA-matched budget
    `rank * (in + out)` -- but writes the new indices only into the python dict, leaving the
    registered buffer `qwha_indices_default` holding the original randperm. The forward reads the
    dict, so training is correct; `save_pretrained` reads the buffer, so a trained checkpoint would
    ship indices that do not match its spectrum. `update_indices()` writes both.

Nothing here changes the method's arithmetic.
"""

import json
import os
import sys

import torch

QWHA_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQAT_REPO = os.path.dirname(os.path.dirname(QWHA_REPO))

# Upstream's own imports are written for `python src/init/initialize.py` run from src/init, so both
# directories go on the path before anything upstream is imported. SQAT_REPO joins them so
# `src.gptq` (which uses relative imports, hence must come in as part of the package) resolves
# from any cwd -- make_bcal_base.py chdirs to the repo root for the data paths, which is not the
# same thing as being importable from it.
for _p in (os.path.join(QWHA_REPO, "src"), os.path.join(QWHA_REPO, "src", "init"), SQAT_REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

QWHA_CACHE_PATH = os.getenv(
    "QWHA_CACHE_PATH", "/scratch/users/nus/jingzege/SQAT_outputs/qwha_cache"
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]


# ---------------------------------------------------------------------------- paths

def gptq_base_dir(model_id: str, bits: int, group_size: int) -> str:
    """Where the plain-GPTQ base for one (bits, group_size) cell lives.

    Upstream's cache name is `{model_id}-{bits}bits-g{group_size}`; kept byte-identical so
    upstream's own loader finds the same directory.
    """
    return os.path.join(QWHA_CACHE_PATH, "gptq_models", f"{model_id}-{bits}bits-g{group_size}")


def init_ckpt_dir(model_id: str, bits: int, group_size: int, rank: int) -> str:
    """Where the AdaAlloc-initialized (untrained) adapter for one cell lives."""
    return os.path.join(
        QWHA_CACHE_PATH,
        "initialized_checkpoints",
        f"{model_id}-{bits}bit-gptq-qwha-rank{rank}-g{group_size}",
    )


# ---------------------------------------------------------------------------- model

def build_qwha_model(
    gptq_path: str,
    *,
    rank: int = 64,
    scale: float = 4000.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    random_loc_seed: int = 777,
):
    """Quantized base + QWHA adapter, on ONE device.

    Mirrors `utils.get_quantized_peft_model` step for step (same QWHAConfig, same per-layer
    budget `rank * (in + out)`, same fp32 spectrum on top of a bf16 base, same seeded randperm),
    with the path/device/buffer fixes described in this module's docstring.
    """
    from transformers import AutoModelForCausalLM
    from peft import QWHAConfig, TaskType, get_peft_model

    if not os.path.isdir(gptq_path):
        raise FileNotFoundError(
            f"No quantized base at {gptq_path}. Run sqat/quantize_base.py for this cell first."
        )

    quantized = AutoModelForCausalLM.from_pretrained(gptq_path, device_map={"": device})

    # Upstream's flat 524288 is a placeholder: every layer's spectrum is rebuilt just below at the
    # LoRA-matched budget rank*(in+out). It still has to clear peft's `n_frequency <= in*out`
    # check, which every Llama-2-7B projection does and a tiny smoke model does not -- so the
    # placeholder is capped at the smallest adapted layer instead of being hard-coded.
    smallest = min((m.in_features * m.out_features
                    for n, m in quantized.named_modules()
                    if hasattr(m, "in_features") and any(n.endswith(t) for t in TARGET_MODULES)),
                   default=524288)
    peft_config = QWHAConfig(
        task_type=TaskType.CAUSAL_LM,
        n_frequency=min(524288, smallest),   # placeholder; overwritten per layer just below
        target_modules=TARGET_MODULES,
        scaling=scale,
        random_loc_seed=random_loc_seed,
        init_weights=True,           # zero spectrum
    )
    model = get_peft_model(quantized, peft_config)
    model.to(dtype)

    for _, module in model.named_modules():
        if hasattr(module, "qwha_spectrum"):
            n_frequency = rank * (module.in_features + module.out_features)
            module.qwha_n_frequency["default"] = n_frequency
            # fp32 spectrum on a bf16 base -- upstream does the same, after the .to(dtype).
            module.qwha_spectrum["default"] = torch.nn.Parameter(
                torch.zeros(n_frequency, device=device), requires_grad=True
            )
            indices = torch.randperm(
                module.out_features * module.in_features,
                generator=torch.Generator().manual_seed(module.qwha_random_loc_seed["default"]),
            )[:n_frequency]
            indices = torch.stack([indices // module.in_features,
                                   indices % module.in_features], dim=0)
            module.update_indices("default", indices)     # dict AND registered buffer

    for _, module in model.named_modules():
        if hasattr(module, "qweight"):   # gptqmodel QuantLinear: these two never follow .to()
            module.wf_unsqueeze_zero = module.wf_unsqueeze_zero.to(device)
            module.wf_unsqueeze_neg_one = module.wf_unsqueeze_neg_one.to(device)

    return model.to(device)


def load_qwha_adapter(model, path: str, *, scale: float):
    """Load spectrum + indices from an adapter checkpoint into `model`, at scaling `scale`.

    Same rescale as `utils.load_from_checkpoint` (a checkpoint written at scaling s0 is replayed
    at scaling s by multiplying the spectrum by s0/s, which leaves delta W unchanged), plus the
    index-buffer fix.
    """
    from safetensors.torch import load_file

    ckpt = load_file(os.path.join(path, "adapter_model.safetensors"))
    with open(os.path.join(path, "adapter_config.json")) as f:
        ckpt_scaling = json.load(f)["scaling"]

    loaded = 0
    for name, module in model.named_modules():
        if not hasattr(module, "qwha_spectrum"):
            continue
        spectrum = ckpt[f"{name}.qwha_spectrum"].to(model.device)
        indices = ckpt[f"{name}.qwha_indices_default"].to(model.device)
        module.qwha_spectrum["default"] = torch.nn.Parameter(
            spectrum.to(torch.float32) * (ckpt_scaling / scale), requires_grad=True
        )
        module.update_indices("default", indices)
        module.qwha_n_frequency["default"] = spectrum.numel()
        module.qwha_scaling["default"] = scale
        loaded += 1
    if loaded == 0:
        raise RuntimeError(f"No QWHA layers were loaded from {path}")
    print(f"[QWHA] loaded {loaded} adapter layers from {path} "
          f"(checkpoint scaling {ckpt_scaling} -> {scale})")
    return model


# ---------------------------------------------------------------------------- data

_SQAT_DATA = None


def sqat_data_module():
    """This repo's `src/data.py`, loaded by path and cached.

    By path, not as `src.data`: importing the package would run src/__init__.py, which pulls
    model_loader (bitsandbytes) and trainer -- neither belongs in the QWHA env. src/data.py
    itself has no intra-repo imports. It is the single source of the cell's prompt, EOS
    convention, `loss_span` masking, collator and calibration sampling, shared by the training
    set, the GPTQ base's calibration and the AdaAlloc X^T X.
    """
    global _SQAT_DATA
    if _SQAT_DATA is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "sqat_data", os.path.join(SQAT_REPO, "src", "data.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SQAT_DATA = module
    return _SQAT_DATA


def sqat_gptq_module():
    """This repo's `src/gptq.py`, without running `src/__init__.py`.

    src/gptq.py uses relative imports, so it has to come in as part of a `src` package -- but the
    real package's __init__ pulls model_loader (bitsandbytes) and trainer, neither of which
    belongs in the QWHA env. A synthetic package with just the four modules of gptq's dependency
    chain (qat_base <- quant_primitives, permute_common, gptq -- all of which need only torch and
    tqdm) gives the identical code with none of that.
    """
    if "src.gptq" in sys.modules:
        return sys.modules["src.gptq"]

    import importlib.util
    import types

    pkg = sys.modules.get("src")
    if pkg is None:
        pkg = types.ModuleType("src")
        pkg.__path__ = [os.path.join(SQAT_REPO, "src")]
        sys.modules["src"] = pkg
    for name in ("qat_base", "quant_primitives", "permute_common", "gptq"):
        spec = importlib.util.spec_from_file_location(
            f"src.{name}", os.path.join(SQAT_REPO, "src", f"{name}.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"src.{name}"] = module
        spec.loader.exec_module(module)
    return sys.modules["src.gptq"]


def load_sqat_data_module(cfg: dict, tokenizer):
    """Tokenized train set + collator, straight out of this repo's `src/data.py`.

    The QWHA row has to sit in the same experimental cell as the SALT-Q and QA-LoRA rows, and the
    cell is defined by the prompt, the EOS convention and `data.loss_span` -- so the tokenization
    comes from the same module those rows used, not from a copy.
    """
    data = sqat_data_module()
    train_dataset, eval_dataset = data.load_dataset_for_training(cfg, tokenizer)
    return train_dataset, eval_dataset, data.build_data_collator(tokenizer)
