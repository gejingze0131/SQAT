"""
Model loading: BitsAndBytes NF4/NF3 quantization + PEFT LoRA.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


def _get_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    """Build BitsAndBytesConfig for 3-bit or 4-bit quantization."""
    bits = cfg["model"]["quant_bits"]
    compute_dtype = getattr(torch, cfg["model"]["dtype"])

    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["model"]["quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg["model"]["double_quant"],
        )
    elif bits in (2, 3):
        # bitsandbytes has no native 2-/3-bit. Strategy: load the QLoRA base in 4-bit NF4 and
        # apply the actual 2-/3-bit rounding downstream — in the QAT fakequant (salient slice)
        # and at export (salient = canonical grid, non-salient = GPTQ). For sqat_permute the
        # permuted base is the NF4 base; quant_bits only sets the fakequant/export target.
        print(f"[WARN] {bits}-bit requested. Using 4-bit NF4 as the QLoRA base; "
              f"{bits}-bit rounding is applied in the QAT/export quantizer.")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg["model"]["double_quant"],
        )
    else:
        raise ValueError(f"Unsupported quant_bits={bits}. Use 2, 3 or 4.")


def _get_lora_config(cfg: dict) -> LoraConfig:
    """Build PEFT LoRA config."""
    lora_cfg = cfg["lora"]
    return LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(cfg: dict):
    """
    Load quantized model with LoRA adapters.
    
    Returns:
        model: PeftModel (ready for training)
        tokenizer: AutoTokenizer
        base_model_ref: reference to the underlying quantized model (for QAT hooks)
    """
    model_name = cfg["model"]["name"]

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Quantized base model ---
    bnb_config = _get_bnb_config(cfg)

    attn_impl = cfg["model"].get("attn_implementation", None)
    model_kwargs = dict(
        pretrained_model_name_or_path=model_name,
        quantization_config=bnb_config,
        # device_map="auto",  # accelerate will handle distribution
        dtype=getattr(torch, cfg["model"]["dtype"]),
        trust_remote_code=True,
    )
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Prepare for k-bit training (freeze base, enable gradient checkpointing, etc.)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Keep a reference to the unwrapped base model before PEFT wrapping
    base_model_ref = model

    # --- LoRA ---
    lora_config = _get_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer, base_model_ref
