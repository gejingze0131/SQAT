"""
QEFT weak-column tuning on Commonsense-170k, in THIS repo's cell.

Method side (QEFT, EMNLP 2024 Findings): the fp16 weak columns of every target linear are the
ONLY trainable tensors; the INT-b bulk is frozen and there is no adapter. AdamW, no dropout, no
regularizer — upstream has none either (its Limitations section says so).

Recipe side (this repo, so the row is comparable to SALT-Q / QA-LoRA / QWHA):
datasets/commonsense/train.json, the repo PROMPT, data.loss_span = instruction+response,
1 epoch, effective batch 80, max_seq_len 2048, cosine + 0.03 warmup, wd 0.01, max_grad_norm 0.3,
bf16, group_by_length false, seed 42, gradient checkpointing on; tokenization imported from
src/data.py rather than copied.

The learning rate is QEFT's own (paper Table 4 / Table 7), not this repo's — a baseline retuned
to match the method under test is not a baseline. See the config header for the exact value and
where it comes from.

Checkpoints hold ONLY the weak columns; the frozen base stays in qeft.base_dir and is referenced
by pointer (rebuilding it would move the weak-column SET and orphan every saved tensor).

    torchrun --nproc_per_node 4 baseline/QEFT/sqat/train_commonsense.py --config <cfg>
"""

import argparse
import dataclasses
import json
import os
import sys

import torch
import yaml
from transformers import Trainer, TrainingArguments, set_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qeft_common import (WEAK_PARAM_NAME, build_qeft_model, load_qeft_trainable,  # noqa: E402
                         load_sqat_data_module, save_qeft_trainable)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.model_loader import load_tokenizer  # noqa: E402


def _make_qeft_trainer_cls(base_dir: str):
    """Trainer that saves (and resumes from) only the trained weak columns.

    HF's default would write the whole 13 GB dense model on every save_steps, of which all but
    ~0.7 GB is the frozen base that already exists on disk.
    """

    class _QEFTTrainer(Trainer):
        def _save(self, output_dir: str = None, state_dict=None):
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            model = self.accelerator.unwrap_model(self.model)
            save_qeft_trainable(model, output_dir, base_dir=base_dir)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            load_qeft_trainable(
                self.accelerator.unwrap_model(self.model if model is None else model),
                resume_from_checkpoint)

    return _QEFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--base_dir", default=None)
    ap.add_argument("--resume_from", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    qcfg, tcfg, mcfg = cfg["qeft"], cfg["training"], cfg["model"]
    if args.output_dir:
        tcfg["output_dir"] = args.output_dir
    if args.learning_rate:
        tcfg["learning_rate"] = args.learning_rate
    base_dir = args.base_dir or qcfg["base_dir"]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    set_seed(tcfg["seed"])

    dtype = getattr(torch, mcfg.get("dtype", "bfloat16"))
    tokenizer = load_tokenizer(cfg)
    zp_lr = float(tcfg.get("zp_lr", 0.0))
    model, meta = build_qeft_model(base_dir, dtype=dtype, device=device, train_zp=zp_lr > 0)
    model.config.use_cache = False

    eff_batch = (tcfg["per_device_train_batch_size"] * tcfg["gradient_accumulation_steps"]
                 * world_size)
    if local_rank == 0:
        print("=" * 78)
        print(f"  QEFT — weak column tuning | INT{meta['q_bits']} g{meta['group_size']} "
              f"k={meta['k']} | fp16 share {meta['fp16_share'] * 100:.2f}% "
              f"({meta['effective_bits']:.2f} effective bits)")
        print(f"  base   {base_dir}")
        print(f"  lr     {tcfg['learning_rate']}   eff batch {eff_batch}   "
              f"epochs {tcfg['num_epochs']}")
        if zp_lr > 0:
            n_zp = sum(p.numel() for n, p in model.named_parameters()
                       if p.requires_grad and n.endswith('zp_shift'))
            print(f"  zp     TRAINABLE non-weak zero-points, {n_zp / 1e6:.1f}M shifts at "
                  f"zp_lr {zp_lr} (LEVEL units, via the recovered per-group step)")
        print(f"  out    {tcfg['output_dir']}")
        print("=" * 78)

    if tcfg.get("gradient_checkpointing", True):
        # The weak columns live inside the blocks, so without this the first block's recomputed
        # graph has no input that requires grad and its checkpoint segment produces no gradient.
        model.enable_input_require_grads()

    # The length-grouping flag was renamed across transformers versions: v4 takes the bool
    # `group_by_length`, v5 takes `train_sampling_strategy="group_by_length"|"random"`. Detect
    # which the installed version exposes, exactly as src/trainer.py does, so this script runs
    # unchanged on either. (These rows set it to false: the cell fixes the sampling order.)
    _ta_fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    _gbl = bool(tcfg.get("group_by_length", False))
    _length_kwargs = {}
    if "group_by_length" in _ta_fields:
        _length_kwargs["group_by_length"] = _gbl
    elif "train_sampling_strategy" in _ta_fields:
        _length_kwargs["train_sampling_strategy"] = "group_by_length" if _gbl else "random"

    training_args = TrainingArguments(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg["num_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        max_grad_norm=tcfg["max_grad_norm"],
        bf16=tcfg["bf16"],
        fp16=tcfg["fp16"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        dataloader_num_workers=tcfg["dataloader_num_workers"],
        report_to=tcfg["report_to"],
        seed=tcfg["seed"],
        remove_unused_columns=False,
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Every rank holds an identical copy of the frozen base; nothing in it ever changes.
        ddp_broadcast_buffers=False,
        ddp_find_unused_parameters=False,
        **_length_kwargs,
    )

    # One rank tokenizes, the rest read its cache — 147k records x 4 processes otherwise.
    with training_args.main_process_first(desc="dataset"):
        train_dataset, _eval_dataset, collator = load_sqat_data_module(cfg, tokenizer)

    trainer_cls = _make_qeft_trainer_cls(base_dir)
    # zp_lr > 0: the two tiers live in different units (weight units vs quantization LEVELS), so
    # they need their own optimizer groups — exactly src/trainer.py's reasoning for SALT-Q.
    _opt = None
    if zp_lr > 0:
        _weak = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith(WEAK_PARAM_NAME)]
        _zps = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith("zp_shift")]
        assert _weak and _zps
        _opt = torch.optim.AdamW(
            [{"params": _weak, "lr": tcfg["learning_rate"], "weight_decay": tcfg["weight_decay"]},
             {"params": _zps, "lr": zp_lr, "weight_decay": 0.0}],
            lr=tcfg["learning_rate"], betas=(0.9, 0.999))
    trainer = trainer_cls(model=model, args=training_args, train_dataset=train_dataset,
                          data_collator=collator, processing_class=tokenizer,
                          optimizers=(_opt, None))
    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_state()

    final_dir = os.path.join(tcfg["output_dir"], "final")
    if local_rank == 0:
        save_qeft_trainable(trainer.accelerator.unwrap_model(model), final_dir, base_dir=base_dir)
        meta_out = dict(
            model_id=mcfg["name"], bits=meta["q_bits"], group_size=meta["group_size"],
            k=meta["k"], oproj_weak=meta["oproj_weak"], base_dir=os.path.abspath(base_dir),
            fp16_share=meta["fp16_share"], effective_bits=meta["effective_bits"],
            learning_rate=tcfg["learning_rate"], zp_lr=zp_lr, epochs=tcfg["num_epochs"],
            effective_batch=eff_batch,
            # data.loss_span does not exist for a raw-text causal-LM dataset (task_type: lm):
            # there is no prompt to mask, every token is supervised. A bare subscript here threw
            # KeyError AFTER the WikiText-2 run had finished all 528 steps and saved its weak
            # columns, losing only this sidecar (2026-09-03, job 16072464). Record what the run
            # actually supervised instead of assuming the instruction schema.
            loss_span=cfg["data"].get(
                "loss_span",
                f"n/a (task_type={cfg['data'].get('task_type', 'instruction')}: every token supervised)"),
            task_type=cfg["data"].get("task_type", "instruction"),
            calibration=cfg["qat"]["sqat"], config=os.path.abspath(args.config),
            trainable_params=int(sum(p.numel() for n, p in model.named_parameters()
                                     if p.requires_grad and WEAK_PARAM_NAME in n)),
        )
        with open(os.path.join(final_dir, "qeft_run_meta.json"), "w") as f:
            json.dump(meta_out, f, indent=2)
        print(f"[QEFT] peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        print(f"[QEFT] saved trained weak columns to {final_dir}")


if __name__ == "__main__":
    main()
