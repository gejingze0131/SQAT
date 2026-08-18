#!/usr/bin/env python
"""
Localize a suspicious initial training loss.

The epoch-1 commonsense run opened at 8.14 (the mean over steps 1-10, so step 1 was higher
still) with grad_norm 420. Two very different things could produce that, and they call for
opposite responses:

  * response-only supervision. Only ~8 tokens per example carry loss, all of them the answer,
    and the model has not yet seen the "### Response:" convention. This RAISES the number
    without anything being wrong.
  * a damaged starting model. INT3 codes, the segment permutation, the AWQ fold or the
    (s, z) initialization could have left the base far from the fp16 model it should
    reproduce, in which case the run is measuring recovery from damage, not fine-tuning.

Separating them needs three models on identical batches, and both loss definitions on each:

  1. meta-llama/Llama-2-7b-hf                fp16 reference
  2. the permuted fp16 base + boundary gathers   isolates permutation / AWQ folding
  3. the SALT-Q base, UNTRAINED                  the actual step-0 state

The gap 1 -> 2 is the offline transforms. The gap 2 -> 3 is quantization. The gap between the
full-sequence and response-only columns is what the masking alone costs. `python
scripts/diagnose_init_loss.py --config <cfg> --saltq_base_dir <dir>` prints all six numbers.
"""

import argparse
import gc
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import build_data_collator, load_dataset_for_training
from src.model_loader import load_tokenizer


@torch.no_grad()
def mean_losses(model, batches, device):
    """Return (response_only_loss, full_sequence_loss) averaged over batches."""
    model.eval()
    resp_tot = resp_n = full_tot = full_n = 0.0
    for batch in batches:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits.float()

        shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
        for name, lab in (("resp", labels), ("full", input_ids.masked_fill(~attn, -100))):
            shift_lab = lab[:, 1:].reshape(-1)
            n = int((shift_lab != -100).sum())
            if n == 0:
                continue
            loss = torch.nn.functional.cross_entropy(
                shift_logits, shift_lab, ignore_index=-100, reduction="sum")
            if name == "resp":
                resp_tot += loss.item(); resp_n += n
            else:
                full_tot += loss.item(); full_n += n
        del logits
    return resp_tot / max(resp_n, 1), full_tot / max(full_n, 1)


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--saltq_base_dir", required=True)
    ap.add_argument("--permuted_base_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda:0")
    dtype = getattr(torch, cfg["model"]["dtype"])

    # The SALT-Q tokenizer comes from the frozen-code base, the same one training used.
    tokenizer = load_tokenizer(cfg, name=args.saltq_base_dir)
    cfg["data"]["max_train_samples"] = args.n_samples
    cfg["data"]["num_proc"] = 4
    train_ds, _ = load_dataset_for_training(cfg, tokenizer)
    collator = build_data_collator(tokenizer)
    batches = list(DataLoader(train_ds, batch_size=args.batch_size,
                              collate_fn=collator, shuffle=False))

    supervised = sum(int((b["labels"] != -100).sum()) for b in batches)
    total = sum(int(b["attention_mask"].sum()) for b in batches)
    print(f"\n[data] {len(train_ds)} samples | {total} real tokens | "
          f"{supervised} supervised ({100*supervised/total:.1f}%)\n")

    rows = []
    from transformers import AutoModelForCausalLM

    print("[1/3] fp16 reference: " + cfg["model"]["name"])
    m = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], dtype=dtype, low_cpu_mem_usage=True).to(device)
    rows.append(("fp16 reference", *mean_losses(m, batches, device)))
    free(m)

    print("[2/3] permuted fp16 base + boundary gathers")
    from src.permute_common import register_boundary_gathers_from_meta, PERM_META_FILENAME
    m = AutoModelForCausalLM.from_pretrained(
        args.permuted_base_dir, dtype=dtype, low_cpu_mem_usage=True).to(device)
    register_boundary_gathers_from_meta(m, os.path.join(args.permuted_base_dir, PERM_META_FILENAME))
    rows.append(("permuted fp16 base", *mean_losses(m, batches, device)))
    free(m)

    print("[3/4] SALT-Q base, UNTRAINED, NO boundary gathers  (a control, not a valid model)")
    from src.qat_saltq import build_saltq_model
    sq = cfg["qat"].get("saltq", {}) or {}
    m, saltq_meta = build_saltq_model(
        args.saltq_base_dir, dtype=dtype, param_dtype=torch.float32,
        gradient_checkpointing=False,
        train_layernorms=bool(sq.get("train_layernorms", False)),
        train_scale=bool(sq.get("train_scale", False)),
        continuous_z=bool(sq.get("continuous_z", True)),
    )
    m = m.to(device)
    # build_saltq_model does NOT register the gathers — SALTQHandler.prepare_model does, which
    # is what training calls. Measuring here first makes the cost of forgetting them explicit,
    # because forgetting them looks exactly like catastrophic quantization damage.
    rows.append(("SALT-Q base, NO gathers", *mean_losses(m, batches, device)))

    print("[4/4] SALT-Q base, UNTRAINED, gathers registered  (this is what step 0 sees)")
    register_boundary_gathers_from_meta(m, saltq_meta["perm_meta"])
    rows.append(("SALT-Q base (untrained)", *mean_losses(m, batches, device)))
    free(m)

    print(f"\n{'model':<28}{'response-only':>15}{'full-sequence':>15}")
    print("-" * 58)
    for name, resp, full in rows:
        print(f"{name:<28}{resp:>15.4f}{full:>15.4f}")
    print("-" * 58)
    print(f"masking alone (fp16):            +{rows[0][1] - rows[0][2]:.4f}")
    print(f"permutation/AWQ  (1 -> 2):       {rows[1][1] - rows[0][1]:+.4f} response-only")
    print(f"quantization     (2 -> 4):       {rows[3][1] - rows[1][1]:+.4f} response-only")
    print(f"missing gathers  (4 -> 3):       {rows[2][1] - rows[3][1]:+.4f} response-only  (control)")
    print(f"\nTraining opened at 8.14 (mean of steps 1-10).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
