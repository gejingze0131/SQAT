"""
Training loop builder using HuggingFace Trainer + QAT callbacks.
"""

import math
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


class SALTQAdamW(torch.optim.Optimizer):
    """AdamW with an optional PER-TENSOR second moment, selected per param group.

    WHY. The zero-point gradient is dL/dC = -gy^T @ pool_g(x): a sum of T outer products, so it is
    NATURALLY low rank. Measured on the z-only INT2 g32 run's own optimizer state (40 projections,
    [11008, 128], scripts/measure_adam_conditioning.py + the spectrum probe):

        saltq_z shape   sr(gradient) top-1    sr(applied) top-1    PAIRED flattening
        [4096, 128]        1.48      68.5%       4.48     26.4%          3.24x
        [11008, 128]       3.48      29.4%       6.24     16.1%          1.56x
        [4096, 344]        4.06      26.4%       6.00     18.0%          1.17x
                                                     median over tensors: 2.92x

    Pair inside each tensor and bucket by shape: a ratio of two MARGINAL medians is not a
    flattening ratio, and SALT-Q's zero-point optimizer group holds saltq_z and the salient
    slice's lsq_w_zp together, so anything pooled over it mixes two unrelated objects.

    Per-COORDINATE normalisation drives every coordinate toward the same step size, i.e. toward a
    FLAT spectrum. That is the whole failure: QA-LoRA's successful update has stable rank 1.62 /
    top-1 61.9% -- statistically the same shape as z's own gradient -- while SALT-Q's realised
    update came out at 5.67 / 17.7%. QA-LoRA is immune not because of its optimizer but because
    B@A is rank-constrained BY CONSTRUCTION: Adam may flatten B and A entrywise all it likes, the
    product is still rank <= 64 and in practice 1.6.

    Raising zp_lr cannot fix this. Matching QA-LoRA's dominant singular direction needs x4.73,
    which is the x5 that was already run and lost 27.5 points, because the same scalar also
    multiplies the ~5 directions that carry only noise.

    WHAT per_tensor_v=True DOES. Keep ONE scalar v per parameter tensor, the EMA of mean(g^2),
    instead of one per coordinate:

        per-coordinate   u_i = m_i / (sqrt(v_i) + eps)
        per-tensor       u_i = m_i / (sqrt(mean_j v_j) + eps)

    The denominator is now constant WITHIN a tensor, so the update keeps the gradient's relative
    structure -- its concentration survives -- while still being normalised per tensor, so the
    optimizer stays scale-free across tensors and across training phases. That last property is
    what a plain SGD group would give up: z's gradient scale moves ~10x between the plateau and
    after the escape (logged grad_norm p50 0.144 vs 1.557), and ~95x between the z-only and
    train_salient configurations, so a fixed SGD lr cannot serve both.

    Side effect worth having: the z tier's exp_avg_sq drops from 202M floats to 224 scalars,
    about 810 MB per rank.

    Groups WITHOUT per_tensor_v take the ordinary AdamW path, so this class is a drop-in and the
    default configuration is mathematically identical to what ran before.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 per_tensor_v=False):
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, per_tensor_v=per_tensor_v))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            per_tensor = bool(group.get("per_tensor_v", False))

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SALTQAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    # A 0-dim tensor, not a python float: it has to live on the parameter's device
                    # and be carried by state_dict() across save/resume like any other state.
                    state["exp_avg_sq"] = (torch.zeros((), dtype=torch.float32, device=p.device)
                                           if per_tensor else torch.zeros_like(p))

                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                if per_tensor:
                    # mean over the WHOLE tensor -> the denominator is constant within it, which
                    # is precisely what leaves the update's spectrum alone.
                    v.mul_(beta2).add_(grad.float().pow(2).mean(), alpha=1.0 - beta2)
                else:
                    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)                      # decoupled, as in AdamW

                bc1 = 1.0 - beta1 ** t
                bc2_sqrt = math.sqrt(1.0 - beta2 ** t)
                denom = v.sqrt().div_(bc2_sqrt).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)          # denom broadcasts when 0-dim

        return loss


def saltq_lr(cfg: dict, key: str, fallback: float) -> float:
    """Resolve one per-bit SALT-Q hyperparameter for the bit width actually being trained.

    Named for its first use (the three learning rates) but the body is generic — `zp_eps` goes
    through it too, for the same reason: it is an absolute quantity whose right value depends on
    the unit its parameter tier lives in, and that unit is a function of the bit width.

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
            # 1e-8 == HF's adam_epsilon == the previous behaviour. Lowering it is opt-in per
            # config so it costs exactly one variable; see _make_saltq_trainer_cls's docstring.
            zp_eps=saltq_lr(cfg, "zp_eps", 1e-8),
            # false == the ordinary per-coordinate Adam every run so far used.
            zp_per_tensor_v=bool(sq_cfg.get("zp_per_tensor_v", False)),
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
                            saltq_base_dir: str = None, zp_eps: float = 1e-8,
                            zp_per_tensor_v: bool = False):
    """
    Trainer subclass for SALT-Q.

    create_optimizer — three param groups matching the method's three tiers of freedom:
      1. `weight_salient`                   real weights, weight units, `salient_lr`
      2. `lsq_w_scale` / `saltq_s`          scales, weight units (~1e-2), `scales_lr`, no decay
      3. `lsq_w_zp` / `saltq_z`             zero-points, QUANTIZATION LEVEL units (step 1.0),
                                            `zp_lr`, no decay, `zp_eps`
      4. everything else trainable (e.g. LayerNorms if opted in), base lr

    Groups 2 and 3 used to share one lr. They must not: a zero-point measured in levels needs an
    lr two to three orders of magnitude larger than a scale measured in weight units, and sharing
    the scale's lr left every zero-point in the model at 0.0000% change over a full run.

    THE SAME ARGUMENT APPLIES TO ADAM'S eps, AND USED NOT TO BE APPLIED.
    configs/saltq*.yaml derive every lr from "under Adam the update is lr * m/(sqrt(v)+eps), so a
    constant factor on the gradient CANCELS — only lr matters". That holds exactly when
    eps << sqrt(v). But eps is an ABSOLUTE floor while each tier's gradient magnitude is set by the
    unit it lives in: dL/dz = -s * sum_g(dL/dW) is numerically ~1e-8 at INT2 g32 precisely because
    z is measured in LEVELS. So the assumption fails for the one tier that derivation calls "the
    entire freedom of the non-salient 98%". Measured on the INT2 g32 anchor's own optimizer state
    (scripts/measure_adam_conditioning.py, checkpoint-1845):

        tier              sqrt(v) p50   % below eps=1e-8   sqrt(v)/(sqrt(v)+eps) at p50 / p10
        salient weights     6.945e-7           7.7%                 0.986 / 0.741
        LSQ scales          2.880e-7           2.8%                 0.966 / 0.808
        ZERO-POINTS         2.784e-8          31.5%                 0.736 / 0.150
        (QA-LoRA's adapter  8.233e-7           0.8%                 0.988 / 0.915)

    A third of all zero-points sit below the eps floor; the median loses 26% of its Adam step and
    the bottom decile 85%. No other tier in either method is within an order of magnitude of that.
    And because v is an EMA with beta2=0.999 read off the END of an 1845-step epoch, it is
    dominated by the post-escape phase where gradients are ~10x larger than on the plateau (logged
    grad_norm p50 0.144 on the plateau vs 1.557 late) — so those numbers are a LOWER BOUND on the
    damping during the plateau, which is when it matters.

    This also undercuts the transportability of the displacement constant c ~= 0.47*sqrt(T) the
    configs use to predict |Δz| from zp_lr: c only carries across datasets while sqrt(v) >> eps in
    both. It was measured on MetaMath, and a MetaMath |Δz| target has now missed on
    Commonsense-170k three times (0.039 and 0.0143 levels realized against a 0.1-0.3 band).

    LOWERING IT WAS TRIED AND IT FAILED. configs/saltq_cs170k_int2_g32_ep1_zpeps.yaml took the
    anchor to zp_eps 1e-10 and never left the plateau: final loss 0.1286, MEAN(7) 38.57 with 7 of
    7 tasks below their majority class. That is the SAME failure as zp_lr x5 (0.1287, 38.28, the
    same 7/7 degenerate), which is the tell: both knobs do the same thing.

    The measurement above is real; the DIRECTION read off it was wrong. eps sitting at 26% of the
    denominator was not only damping the tier, it was partially DE-NORMALISING it — pushing
    m/(sqrt(v)+eps) away from Adam and toward SGD, which is the one thing this tier needs. The
    real defect is that per-coordinate normalisation destroys the gradient's concentration (see
    SALTQAdamW: the applied update's stable rank is ~2.9x the gradient's), so lowering eps made Adam
    normalise HARDER and moved things the wrong way. The useful direction for this knob is UP, and
    the principled version of "up" is `zp_per_tensor_v`, which changes the denominator's shape
    instead of its size.

    THE DEFAULT IS 1e-8 = HF's adam_epsilon, i.e. exactly the previous behaviour. It is NOT
    lowered by default: that would silently change what every existing config means, including the
    validated INT3 anchor at 81.48. Set `qat.saltq.zp_eps` (or `zp_eps_by_bits`) to spend it as
    one explicit variable of one run. Groups 1 and 2 keep 1e-8 deliberately — at 1.4% and 3.4%
    median damping they are not worth a variable, and moving them would break that INT3 anchor.

    `adam_epsilon` in the yaml would NOT do this: build_trainer's TrainingArguments(...) call never
    passes it, so a config-level value is silently ignored — and it would hit all four groups.

    _save — writes ONLY the trainable tensors. The frozen int8 codes are several GB, never change,
    and live in the saltq_base dir; a checkpoint records a pointer to it instead of a copy.
    """
    from src.qat_saltq import (
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

            salient = [p for n, p in named.items() if is_salient(n)]
            scales = [p for n, p in named.items() if is_scale(n)]
            zps = [p for n, p in named.items() if is_zp(n)]
            other = [p for n, p in named.items()
                     if not is_salient(n) and not is_scale(n) and not is_zp(n)]

            param_groups = [
                {"params": salient, "weight_decay": self.args.weight_decay, "lr": salient_lr},
                {"params": scales, "weight_decay": 0.0, "lr": scales_lr},
                # "eps" is a per-group AdamW option exactly like "lr" and "weight_decay" (torch
                # fills missing keys from the optimizer defaults, so an explicit value here beats
                # the adam_epsilon that get_optimizer_cls_and_kwargs passes as the default). See
                # the docstring for why the zero-point tier needs its own.
                {"params": zps, "weight_decay": 0.0, "lr": zp_lr, "eps": zp_eps,
                 "per_tensor_v": zp_per_tensor_v},
                {"params": other, "weight_decay": 0.0},
            ]
            param_groups = [g for g in param_groups if g["params"]]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args, opt_model)
            if zp_per_tensor_v:
                # SALTQAdamW is a plain-python AdamW, so switching to it changes the IMPLEMENTATION
                # for every group, not just the zero-points. That is mathematically a no-op (same
                # decoupled-AdamW update, fp32 either way) and on the z-only cell there is no other
                # group at all, but say so in the log rather than leave it implicit.
                kw = {k: optimizer_kwargs[k] for k in ("lr", "betas", "eps")
                      if k in optimizer_kwargs}
                self.optimizer = SALTQAdamW(param_groups, **kw)
            else:
                # Untouched default path: bit-identical to every run before per_tensor_v existed.
                param_groups = [{k: val for k, val in g.items() if k != "per_tensor_v"}
                                for g in param_groups]
                self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
            _default_eps = optimizer_kwargs.get("eps", self.args.adam_epsilon)
            print(
                f"[Trainer][SALT-Q] optimizer groups (grouped by UNITS, see qat_saltq):\n"
                f"  salient weights  {sum(p.numel() for p in salient) / 1e6:7.1f}M  lr={salient_lr:g}"
                f"   eps={_default_eps:g}   (weight units)\n"
                f"  scales           {sum(p.numel() for p in scales) / 1e6:7.1f}M  lr={scales_lr:g}"
                f"   eps={_default_eps:g}   (weight units, wd=0)\n"
                f"  zero-points      {sum(p.numel() for p in zps) / 1e6:7.1f}M  lr={zp_lr:g}"
                f"   eps={zp_eps:g}   (QUANTIZATION LEVELS, wd=0, second moment="
                f"{'PER-TENSOR' if zp_per_tensor_v else 'per-coordinate'})\n"
                f"  other            {sum(p.numel() for p in other) / 1e6:7.1f}M  "
                f"lr={self.args.learning_rate:g}   eps={_default_eps:g}"
            )
            if zp_per_tensor_v:
                print("[Trainer][SALT-Q] zero-point second moment is PER-TENSOR (SALTQAdamW): one "
                      "scalar v per projection instead of one per coordinate, so the update keeps "
                      "the gradient's spectrum. Measured on z-only INT2, paired per tensor: the "
                      "applied update's stable rank is 2.92x the gradient's (per shape 3.24x / "
                      "1.56x / 1.17x). ALL groups now run SALTQAdamW's "
                      "implementation of AdamW; only this group's second moment differs.")
            # Loud, because it is the whole point of the tier and it is invisible in the loss.
            if zp_eps != _default_eps:
                print(f"[Trainer][SALT-Q] zero-point eps OVERRIDDEN: {_default_eps:g} -> "
                      f"{zp_eps:g}. At INT2 g32 the measured sqrt(v) p50 of this tier is 2.8e-8, "
                      f"so the default floor damped the median Adam step by 26% and the bottom "
                      f"decile by 85% (scripts/measure_adam_conditioning.py).")
            else:
                print(f"[Trainer][SALT-Q] zero-point eps at the shared default {zp_eps:g}. This "
                      f"tier's gradient is ~1e-8 because z is in LEVELS; see the docstring before "
                      f"reading any |dz| number as evidence about zp_lr.")
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
