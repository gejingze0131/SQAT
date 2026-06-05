"""
Training loop builder using HuggingFace Trainer + QAT callbacks.
"""

import os
from typing import Optional

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import PeftModel

from .qat_base import QATHandler


# ============================================================================
# QAT Callback (bridges QATHandler into HF Trainer lifecycle)
# ============================================================================

class QATCallback(TrainerCallback):
    """Injects QAT handler hooks into the HF Trainer lifecycle."""

    def __init__(self, qat_handler: QATHandler):
        self.qat_handler = qat_handler

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.qat_handler.on_train_begin(model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.qat_handler.on_step_end(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self.qat_handler.on_train_end(model)


# ============================================================================
# Build Trainer
# ============================================================================

def build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    cfg: dict,
    qat_handler: QATHandler,
) -> Trainer:
    """
    Construct a HuggingFace Trainer with all config wired up.
    
    Args:
        model:         PeftModel (possibly SQAT-patched)
        tokenizer:     Tokenizer
        train_dataset: Tokenized training dataset
        eval_dataset:  Tokenized eval dataset (or None)
        cfg:           Full experiment config dict
        qat_handler:   QAT handler (NoQAT, FullQAT, or SQAT)
    
    Returns:
        Configured Trainer ready for .train()
    """
    train_cfg = cfg["training"]

    # Suffix output dir with QAT mode and bit width
    qat_mode = cfg["qat"]["mode"]
    bits = cfg["model"]["quant_bits"]
    output_dir = f"{train_cfg['output_dir']}-{bits}bit-{qat_mode}"

    # Fix B: batch samples of similar length together so dynamic padding wastes far
    # less memory/compute. Without it one long sample (MetaMath solutions vary a lot,
    # up to max_seq_len) pads the whole batch and spikes activation memory — the
    # "stable then sudden OOM at step 100+" pattern.
    # The flag was renamed across transformers versions: v4 uses the bool
    # `group_by_length=True`; v5 uses `train_sampling_strategy="group_by_length"`.
    # Detect which the installed version exposes so this works on either.
    import dataclasses
    _ta_fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    _group_by_length = train_cfg.get("group_by_length", True)
    _length_kwargs = {}
    if "group_by_length" in _ta_fields:
        _length_kwargs["group_by_length"] = _group_by_length
    elif "train_sampling_strategy" in _ta_fields:
        _length_kwargs["train_sampling_strategy"] = (
            "group_by_length" if _group_by_length else "random"
        )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_grad_norm=train_cfg["max_grad_norm"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=train_cfg["eval_steps"] if eval_dataset else None,
        save_total_limit=train_cfg["save_total_limit"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        report_to=train_cfg["report_to"],
        seed=train_cfg["seed"],
        remove_unused_columns=False,
        # Distributed
        ddp_find_unused_parameters=False,
        # Gradient checkpointing is already set in model_loader
        # Fix B: length-grouping (see _length_kwargs above; version-dependent key).
        **_length_kwargs,
    )

    # Data collator with left-padding for causal LM
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Build trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[QATCallback(qat_handler)],
    )

    return trainer
