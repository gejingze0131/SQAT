"""
Training loop builder using HuggingFace Trainer + QAT callbacks.
"""

import os
from typing import Optional

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import PeftModel

from .data import build_data_collator
from .qat_base import QATHandler


def saltq_lr(cfg: dict, key: str, fallback: float) -> float:
    """Resolve one SALT-Q learning rate for the bit width actually being trained.

    Every SALT-Q lr is set against the unit its parameter lives in, and two of those units are
    the quantization step s(b) = range / (2^b - 1) — so the right lr is a function of the bit
    width, not a constant (see the derivation in configs/saltq.yaml). Without this, `--bits 2`
    would silently keep the INT3 rates: the salient weights would move 0.25 of a step instead of
    0.5 and half the tier's freedom would go unused, which is a quiet version of exactly the bug
    that cost run1 its non-salient segment.

    Order: `<key>_by_bits[bits]`  >  `<key>`  >  fallback. A CLI override writes the scalar and
    clears the map (scripts/train.py), so it still wins.
    """
    sq = cfg.get("qat", {}).get("saltq", {}) or {}
    by_bits = sq.get(f"{key}_by_bits") or {}
    bits = int(cfg["model"]["quant_bits"])
    if bits in by_bits:
        return float(by_bits[bits])
    if str(bits) in by_bits:                     # YAML keys can arrive as strings
        return float(by_bits[str(bits)])
    if by_bits:
        print(f"[Trainer][SALT-Q] {key}_by_bits has no entry for INT{bits}; "
              f"falling back to the scalar {key}.")
    return float(sq.get(key, fallback))


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
        # Pass the trainer output dir so handlers that self-register extra params (FullQAT LSQ
        # scales) can persist them next to the checkpoint before the injectors are removed.
        self.qat_handler.on_train_end(model, output_dir=getattr(args, "output_dir", None))


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
    saltq_base_dir: str = None,
) -> Trainer:
    """
    Construct a HuggingFace Trainer with all config wired up.
    
    Args:
        model:         PeftModel (possibly SQAT-patched)
        tokenizer:     Tokenizer
        train_dataset: Tokenized training dataset
        eval_dataset:  Tokenized eval dataset (or None)
        cfg:           Full experiment config dict
        qat_handler:   QAT handler (NoQAT, FullQAT, SQAT, SegmentPermutedSelectiveQAT, SALTQ)
        saltq_base_dir: SALT-Q only — the frozen-code base a checkpoint points back to

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
        # SALT-Q registers its FROZEN integer codes as module buffers — several GB per rank, and
        # bit-identical on every rank because they are read from the same frozen-code base file.
        # DDP's default broadcast_buffers=True would re-broadcast all of them on every forward.
        ddp_broadcast_buffers=(False if qat_mode == "saltq" else None),
        # Gradient checkpointing is already set in model_loader
        # Fix B: length-grouping (see _length_kwargs above; version-dependent key).
        **_length_kwargs,
    )

    # Right-padding collator: input_ids padded with pad_token_id, labels with IGNORE_INDEX so
    # the prompt span src/data.preprocess masked stays out of the loss after collation.
    data_collator = build_data_collator(tokenizer)

    trainer_cls = Trainer
    enable_lsq = bool(cfg["qat"].get("lsq", {}).get("enabled", False))
    if qat_mode == "saltq":
        # SALT-Q allocates trainable freedom in three tiers, and they cannot share an lr:
        #   salient WEIGHTS (real weights, not an adapter) — a LoRA-sized lr blows them up;
        #   quantization params (s, z) — a grid, needs a small lr and no weight decay;
        #   anything else opted in (e.g. LayerNorms) — base lr.
        # It also must NOT let HF write a full state_dict: the frozen int8 codes are multiple GB
        # and never change, so checkpoints hold only the trainable tensors.
        sq_cfg = cfg["qat"].get("saltq", {}) or {}
        trainer_cls = _make_saltq_trainer_cls(
            salient_lr=saltq_lr(cfg, "salient_lr", train_cfg["learning_rate"]),
            scales_lr=saltq_lr(cfg, "scales_lr",
                               cfg["qat"].get("lsq", {}).get("scales_lr", 1e-5)),
            zp_lr=saltq_lr(cfg, "zp_lr", 1e-3),
            # Only consulted when zplora_rank > 0. QA-LoRA trains its adapter at 5e-3, and the
            # unit derivation lands in the same place: to move the shared coefficient as far per
            # step as zp_lr 1.73e-3 does, lr_B * ||A_col|| must equal zp_lr, and with r=64 over
            # G=128 groups ||A_col|| = 0.408, giving 4.2e-3. Two independent routes, one answer.
            zplora_lr=saltq_lr(cfg, "zplora_lr", 5e-3),
            saltq_base_dir=saltq_base_dir,
        )
    elif enable_lsq:
        # LSQ scale[/zp] are self-registered nn.Parameters. They MUST get their own optimizer
        # group (small lr, no weight decay) — the weight LR (2e-4) blows up a scale, and weight
        # decay would shrink it toward 0. HF's default create_optimizer groups by decay/no-decay
        # only, so it would (a) lump scales into the main lr and (b) maybe apply decay. Override.
        scales_lr = float(cfg["qat"]["lsq"].get("scales_lr", 1e-5))
        trainer_cls = _make_lsq_trainer_cls(scales_lr)

    # Build trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[QATCallback(qat_handler)],
    )

    # Closed-form output-mean recalibration of the non-salient zero-point (src/recalibration.py).
    # Off (0) by default: every existing config is bit-identical to before. When on, z's
    # drift-compensation job is done by direct computation instead of SGD, so it composes with
    # zp-LoRA (adapter = task adaptation) and with a frozen z, but NOT with a trainable full-rank
    # z: SGD and the callback would then both write the same coefficient toward different targets.
    if qat_mode == "saltq":
        recal_interval = int((cfg["qat"].get("saltq", {}) or {}).get("recal_interval", 0))
        if recal_interval > 0:
            sq = cfg["qat"].get("saltq", {}) or {}
            from .recalibration import SALTQMeanRecalibration
            zp_trainable = any(
                p.requires_grad for n, p in model.named_parameters() if n.endswith("saltq_z"))
            if zp_trainable:
                raise RuntimeError(
                    "[SALT-Q] recal_interval > 0 with a TRAINABLE full-rank z: the optimizer and "
                    "the recalibration callback would both drive the non-salient coefficient. "
                    "Freeze z (zplora_rank > 0, or zp_lr paths off) or set recal_interval: 0.")
            trainer.add_callback(SALTQMeanRecalibration(
                model,
                capture_steps=int(sq.get("recal_capture_steps", 8)),
                interval=recal_interval,
            ))

    return trainer


def _make_lsq_trainer_cls(scales_lr: float):
    """
    Build a Trainer subclass whose create_optimizer puts LSQ scale/zp params (names containing
    'lsq_w_scale'/'lsq_w_zp') into a DEDICATED param group: lr=scales_lr, weight_decay=0. All
    other trainable params keep the standard (decay / no-decay) grouping at the base lr.
    """
    from transformers.trainer_pt_utils import get_parameter_names
    try:
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
    except Exception:  # older transformers
        ALL_LAYERNORM_LAYERS = (torch.nn.LayerNorm,)

    class _LSQTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer

            opt_model = self.model
            decay_params = get_parameter_names(opt_model, list(ALL_LAYERNORM_LAYERS))
            decay_params = [n for n in decay_params if "bias" not in n]

            def is_lsq(n):
                return "lsq_w_scale" in n or "lsq_w_zp" in n

            named = {n: p for n, p in opt_model.named_parameters() if p.requires_grad}
            lsq_names = [n for n in named if is_lsq(n)]

            param_groups = [
                {  # decayed, non-LSQ
                    "params": [p for n, p in named.items() if n in decay_params and not is_lsq(n)],
                    "weight_decay": self.args.weight_decay,
                },
                {  # no-decay, non-LSQ
                    "params": [p for n, p in named.items() if n not in decay_params and not is_lsq(n)],
                    "weight_decay": 0.0,
                },
                {  # LSQ scale/zp — dedicated lr, no decay
                    "params": [named[n] for n in lsq_names],
                    "weight_decay": 0.0,
                    "lr": scales_lr,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args, opt_model)
            self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
            n_lsq = len(param_groups[2]["params"])
            print(f"[Trainer][LSQ] Dedicated LSQ optimizer group: {n_lsq} params "
                  f"(lr={scales_lr:g}, weight_decay=0). "
                  f"{'OK — scales WILL be optimized.' if n_lsq > 0 else 'WARNING: 0 LSQ params found!'}")
            return self.optimizer

    return _LSQTrainer


def _make_saltq_trainer_cls(salient_lr: float, scales_lr: float, zp_lr: float,
                            saltq_base_dir: str = None, zplora_lr: float = 5e-3):
    """
    Trainer subclass for SALT-Q.

    create_optimizer — three param groups matching the method's three tiers of freedom:
      1. `weight_salient`                   real weights, weight units, `salient_lr`
      2. `lsq_w_scale` / `saltq_s`          scales, weight units (~1e-2), `scales_lr`, no decay
      3. `lsq_w_zp` / `saltq_z`             zero-points, QUANTIZATION LEVEL units (step 1.0),
                                            `zp_lr`, no decay
      4. everything else trainable (e.g. LayerNorms if opted in), base lr

    Groups 2 and 3 used to share one lr. They must not: a zero-point measured in levels needs an
    lr two to three orders of magnitude larger than a scale measured in weight units, and sharing
    the scale's lr left every zero-point in the model at 0.0000% change over a full run.

    _save — writes ONLY the trainable tensors. The frozen int8 codes are several GB, never change,
    and live in the saltq_base dir; a checkpoint records a pointer to it instead of a copy.
    """
    from src.qat_saltq import (
        ZPLORA_PARAM_FRAGMENTS,
        SALIENT_WEIGHT_PARAM,
        SCALE_PARAM_FRAGMENTS,
        ZEROPOINT_PARAM_FRAGMENTS,
        load_saltq_trainable,
        save_saltq_trainable,
    )

    class _SALTQTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer

            opt_model = self.model
            named = {n: p for n, p in opt_model.named_parameters() if p.requires_grad}

            def is_scale(n):
                return any(frag in n for frag in SCALE_PARAM_FRAGMENTS)

            def is_zp(n):
                return any(frag in n for frag in ZEROPOINT_PARAM_FRAGMENTS)

            def is_salient(n):
                return SALIENT_WEIGHT_PARAM in n

            def is_zplora(n):
                return any(frag in n for frag in ZPLORA_PARAM_FRAGMENTS)

            salient = [p for n, p in named.items() if is_salient(n)]
            scales = [p for n, p in named.items() if is_scale(n)]
            zps = [p for n, p in named.items() if is_zp(n)]
            zplora = [p for n, p in named.items() if is_zplora(n)]
            other = [p for n, p in named.items()
                     if not is_salient(n) and not is_scale(n) and not is_zp(n)
                     and not is_zplora(n)]

            param_groups = [
                {"params": salient, "weight_decay": self.args.weight_decay, "lr": salient_lr},
                {"params": scales, "weight_decay": 0.0, "lr": scales_lr},
                {"params": zps, "weight_decay": 0.0, "lr": zp_lr},
                # A FOURTH unit: B@A is in level units, so B and A each carry a square root of
                # one. zp_lr cannot be reused for them.
                {"params": zplora, "weight_decay": 0.0, "lr": zplora_lr},
                {"params": other, "weight_decay": 0.0},
            ]
            # A zp-LoRA run must NOT also be training the full-rank z: the adapter REPLACES it.
            # This exact combination ran once, undetected, because build_saltq_model re-enabled
            # saltq_z after SALTQLinear.__init__ froze it. Fail loudly instead.
            if zplora and zps:
                raise RuntimeError(
                    f"[SALT-Q] zp-LoRA is active ({sum(p.numel() for p in zplora)/1e6:.1f}M "
                    f"factors) but the zero-point group is ALSO trainable "
                    f"({sum(p.numel() for p in zps)/1e6:.1f}M params). The adapter replaces the "
                    f"full-rank z; training both means the run is not the experiment its config "
                    f"describes. Check SALTQLinear.param_is_trainable and the freedom-allocation "
                    f"loop in build_saltq_model."
                )

            param_groups = [g for g in param_groups if g["params"]]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args, opt_model)
            self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
            print(
                f"[Trainer][SALT-Q] optimizer groups (grouped by UNITS, see qat_saltq):\n"
                f"  salient weights  {sum(p.numel() for p in salient) / 1e6:7.1f}M  lr={salient_lr:g}"
                f"   (weight units)\n"
                f"  scales           {sum(p.numel() for p in scales) / 1e6:7.1f}M  lr={scales_lr:g}"
                f"   (weight units, wd=0)\n"
                f"  zero-points      {sum(p.numel() for p in zps) / 1e6:7.1f}M  lr={zp_lr:g}"
                f"   (QUANTIZATION LEVELS - needs a far larger lr, wd=0)\n"
                f"  zp-LoRA (B, A)   {sum(p.numel() for p in zplora) / 1e6:7.1f}M  lr={zplora_lr:g}"
                f"   (sqrt of a LEVEL - a fourth unit, wd=0)\n"
                f"  other            {sum(p.numel() for p in other) / 1e6:7.1f}M  "
                f"lr={self.args.learning_rate:g}"
            )
            if not salient:
                print("[Trainer][SALT-Q] WARNING: no salient weight params found!")
            if not zps:
                print("[Trainer][SALT-Q] WARNING: no zero-point params found!")
            return self.optimizer

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            # Mirror of _save: a SALT-Q checkpoint holds only the trainable tensors, so HF's
            # standard weight-file discovery would not find anything to load.
            load_saltq_trainable(
                self.accelerator.unwrap_model(self.model if model is None else model),
                resume_from_checkpoint,
            )

        def _save(self, output_dir: str = None, state_dict=None):
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model)
            save_saltq_trainable(
                model, output_dir, saltq_base_dir=saltq_base_dir,
                learning_rates={"salient_lr": salient_lr, "scales_lr": scales_lr,
                                "zp_lr": zp_lr},
            )
            proc = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
            if proc is not None and hasattr(proc, "save_pretrained"):
                proc.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    return _SALTQTrainer
