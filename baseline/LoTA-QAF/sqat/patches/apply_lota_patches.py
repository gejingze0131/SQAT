"""Apply LoTA-QAF's peft patches to the vendored peft checkout.

The upstream repo ships `LoTA/layer.py`, `LoTA/adapter.py` and `LoTA/lota_merge.py` as
*fragments* to be spliced into installed libraries, not as importable modules: each file's
docstring names the library file it belongs in (`peft/tuners/lora/layer.py`,
`gptqmodel/adapter/adapter.py`) and the surrounding edits are described only in prose
("Related code's processing chain"). This script performs those edits mechanically on
vendor/peft so the training path is the official one and the diff is auditable.

Three edits, all in peft 0.15.1:

  1. tuners/lora/layer.py   append LoTA's IntLinear / triton threshold kernel /
                            ThresholdFunction_pack / pack_bool_tensor / CustomLoraLinear.
                            LoTA_QAF_main.py imports CustomLoraLinear and IntLinear from here.
  2. tuners/lora/model.py   the `_custom_modules` dispatcher must forward
                            `lora_config.custom_config` (LoTA passes residual + threshold
                            through it) — stock peft drops it.
  3. utils/other.py         prepare_model_for_kbit_training must treat the model as GPTQ
                            quantized. A GPTQModel wrapper is not a HF quantized model, so
                            the stock probe returns False and every non-int parameter would
                            be cast to fp32.

Only the gptqmodel INFERENCE adapter (LoTA/adapter.py, LoTA/lota_merge.py) is NOT applied:
this reproduction deploys through export_lota_dense.py instead (see PROVENANCE.md).
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # baseline/LoTA-QAF/sqat/patches
SQAT = HERE.parent                              # our harness
UPSTREAM = SQAT.parent                          # the LoTA-QAF checkout
PEFT = UPSTREAM / "vendor" / "peft" / "src" / "peft"

MARKER = "# ---- LoTA-QAF patch"


def _read(p):
    return p.read_text()


def _write(p, s):
    p.write_text(s)
    print(f"  patched {p}")


def patch_layer():
    """Append LoTA's CustomLoraLinear stack to peft/tuners/lora/layer.py."""
    dst = PEFT / "tuners" / "lora" / "layer.py"
    src = UPSTREAM / "LoTA" / "layer.py"
    body = _read(dst)
    if MARKER in body:
        print("  layer.py already patched")
        return

    frag = _read(src)
    start = frag.index("class IntLinear(nn.Linear):")
    frag = frag[start:]

    body += (
        f"\n\n{MARKER}: LoTA/layer.py, spliced verbatim from the upstream repo ----\n"
        "# CustomLoraLinear replaces the quantized base layer's forward entirely: it decodes the\n"
        "# GPTQ integers, adds the ternary markers thresholded at omega, and folds the residual\n"
        "# into a per-group offset. IntLinear holds the ternary adapter tensors t-SignSGD steps.\n"
        + frag
    )
    _write(dst, body)


def patch_model():
    """Forward lora_config.custom_config into the custom module constructor."""
    dst = PEFT / "tuners" / "lora" / "model.py"
    body = _read(dst)
    if MARKER in body:
        print("  model.py already patched")
        return

    old = (
        "                for key, custom_cls in lora_config._custom_modules.items():\n"
        "                    if isinstance(target_base_layer, key):\n"
        "                        new_module = custom_cls(target, adapter_name, **kwargs)\n"
        "                        break\n"
    )
    new = (
        f"                {MARKER}: pass lora_config.custom_config (residual, threshold) ----\n"
        "                for key, custom_cls in lora_config._custom_modules.items():\n"
        "                    if isinstance(target_base_layer, key):\n"
        "                        custom_params = getattr(lora_config, \"custom_config\", {})\n"
        "                        all_kwargs = {**kwargs, **custom_params}\n"
        "                        new_module = custom_cls(target, adapter_name, **all_kwargs)\n"
        "\n"
        "                        from gptqmodel.nn_modules.qlinear import BaseQuantLinear\n"
        "                        if isinstance(target_base_layer, BaseQuantLinear):\n"
        "                            target.qweight = target_base_layer.qweight\n"
        "                        break\n"
    )
    if old not in body:
        sys.exit("ERROR: model.py dispatcher anchor not found — peft version drifted")
    _write(dst, body.replace(old, new))


def patch_other():
    """prepare_model_for_kbit_training must see the GPTQModel wrapper as GPTQ quantized."""
    dst = PEFT / "utils" / "other.py"
    body = _read(dst)
    if MARKER in body:
        print("  other.py already patched")
        return

    old = '    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"\n'
    new = (
        f"    {MARKER}: a GPTQModel wrapper is not a HF-quantized model, so the stock probe\n"
        "    # returns False and every non-int parameter gets cast to fp32. Upstream\n"
        "    # LoTA_QAF_main.py documents this as a required hardcode.\n"
        "    is_gptq_quantized = True\n"
    )
    if old not in body:
        sys.exit("ERROR: other.py is_gptq_quantized anchor not found — peft version drifted")
    _write(dst, body.replace(old, new))


if __name__ == "__main__":
    if not PEFT.is_dir():
        sys.exit(f"ERROR: vendored peft not found at {PEFT}; run setup_env.sh first")
    print("Applying LoTA-QAF patches to vendor/peft:")
    patch_layer()
    patch_model()
    patch_other()
    print("done.")
