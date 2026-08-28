"""
QWHA on Commonsense-170k, in THIS repo's cell.

Method side (upstream, unchanged): plain-GPTQ INT-b base, AdaAlloc-initialized Walsh-Hadamard
adapter at the LoRA-rank-64 parameter budget, adapter scaling 4000, only the spectrum trainable.

Recipe side (this repo, matching the SALT-Q / QA-LoRA rows it will be tabled against):
datasets/commonsense/train.json, the repo PROMPT, `data.loss_span = instruction+response`,
1 epoch, effective batch 80, cosine + 0.03 warmup, max_grad_norm 0.3, bf16, seed 42, and
tokenization imported from src/data.py rather than copied.

The learning rate is QWHA's own (paper Table 9, Llama-3.1-8B / Alpaca column: 3e-5 at INT3,
2e-5 at INT2 -- the paper has no Llama-2-7B entry). A baseline retuned to match the method under
test is not a baseline.

Launch with torchrun; every rank holds a full copy of the quantized base on its own GPU.
"""

import argparse
import json
import os

import torch
import yaml
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from qwha_common import (build_qwha_model, gptq_base_dir, init_ckpt_dir, load_qwha_adapter,
                         load_sqat_data_module)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--init_ckpt", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    qcfg, tcfg, mcfg = cfg["qwha"], cfg["training"], cfg["model"]
    if args.output_dir:
        tcfg["output_dir"] = args.output_dir
    if args.learning_rate:
        tcfg["learning_rate"] = args.learning_rate

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    set_seed(tcfg["seed"])

    model_id = mcfg["name"]
    bits, gs, rank = mcfg["quant_bits"], qcfg["group_size"], qcfg["rank"]
    gptq_path = gptq_base_dir(model_id, bits, gs)
    init_path = args.init_ckpt or init_ckpt_dir(model_id, bits, gs, rank)

    if local_rank == 0:
        print("=" * 70)
        print(f"  QWHA baseline — INT{bits} g{gs}, rank {rank}, scale {qcfg['scale']}")
        print(f"  base   {gptq_path}")
        print(f"  init   {init_path}")
        print(f"  lr     {tcfg['learning_rate']}   eff batch "
              f"{tcfg['per_device_train_batch_size'] * tcfg['gradient_accumulation_steps'] * world_size}")
        print(f"  out    {tcfg['output_dir']}")
        print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, model_max_length=mcfg["max_seq_len"], padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = build_qwha_model(gptq_path, rank=rank, scale=qcfg["scale"], device=device)
    load_qwha_adapter(model, init_path, scale=qcfg["scale"])
    model.config.use_cache = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if local_rank == 0:
        print(f"[QWHA] trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    if tcfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()

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
        group_by_length=tcfg["group_by_length"],
        dataloader_num_workers=tcfg["dataloader_num_workers"],
        report_to=tcfg["report_to"],
        seed=tcfg["seed"],
        remove_unused_columns=False,
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # The base's packed codes are registered buffers, identical on every rank and several GB
        # of them; DDP's default would re-broadcast the lot on every forward.
        ddp_broadcast_buffers=False,
        ddp_find_unused_parameters=False,
        save_safetensors=True,
    )

    # One rank tokenizes, the rest read its cache -- 147k records x 4 processes otherwise.
    with training_args.main_process_first(desc="dataset"):
        train_dataset, _eval_dataset, collator = load_sqat_data_module(cfg, tokenizer)

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                      data_collator=collator)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=tcfg["output_dir"])

    if local_rank == 0:
        meta = dict(model_id=model_id, bits=bits, group_size=gs, rank=rank,
                    scale=qcfg["scale"], learning_rate=tcfg["learning_rate"],
                    gptq_base=gptq_path, init_ckpt=init_path,
                    loss_span=cfg["data"]["loss_span"],
                    effective_batch=tcfg["per_device_train_batch_size"]
                    * tcfg["gradient_accumulation_steps"] * world_size,
                    epochs=tcfg["num_epochs"], config=os.path.abspath(args.config))
        with open(os.path.join(tcfg["output_dir"], "qwha_run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[QWHA] peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        print(f"[QWHA] saved adapter to {tcfg['output_dir']}")


if __name__ == "__main__":
    main()
