"""LoTA-QAF training on Commonsense-170k, under this repo's training recipe.

This is upstream `LoTA_QAF_main.py` mode 1 (QAF()) with two substitutions and nothing else:

  * the DATA comes from src/data.py -- the same PROMPT, the same loss_span masking, the same
    collator and the same shuffle seed every other row of the tables was trained with.
    Upstream's prepare_dataset() builds alpaca/gsm8k/sql/viggo through a CHAT TEMPLATE, which
    Llama-2-7b-hf (a base model) does not have; keeping it would have compared a different
    prompt as well as a different method.
  * the t-SignSGD sigma schedule gets the run's REAL optimizer-step count. Upstream hardcodes
    it per dataset ({"alpaca": 300, "gsm8k": 117, ...}); with the wrong number the sigma
    anneal (top 5% -> 0.1% over the first 80%, then 0.01%) lands in the wrong place, which is
    the closest thing t-SignSGD has to a learning-rate schedule.

Everything method-side is upstream's: the ternary adapter (peft CustomLoraLinear, patched in
by patches/apply_lota_patches.py), t_signSGD.tSignSGD, omega = 0.75r = 48, sigma_t
0.95/0.999/0.9999, residual offset factor on, max_grad_norm disabled. Those are the paper's
own settings and are NOT retuned to this repo's cell -- a baseline retuned to match the
method under test is not a baseline (same stance as configs/qalora_*.yaml).

Note on `training.learning_rate`: t-SignSGD does not use it. Eq. (6) adds -sign(g) directly,
so lr, warmup and the scheduler only drive an optimizer that never reads them. The key is
kept in the config for the record; the knob that actually controls update strength is
lota.filter_ratio (sigma_t).

    python train_lota.py --config configs/lota_cs170k_int3_g64_ep1_span.yaml \
        --quantized_model_dir outputs/lota_bases/Llama-2-7B_int3_64_asym
"""

import argparse
import math
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.getcwd(), ".triton/cache"))

import torch
import torch._dynamo.config
import yaml

torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.capture_scalar_outputs = True

HERE = os.path.dirname(os.path.abspath(__file__))            # baseline/LoTA-QAF/sqat
UPSTREAM = os.path.abspath(os.path.join(HERE, ".."))         # the LoTA-QAF checkout
SQAT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)          # lota_common
sys.path.insert(0, UPSTREAM)      # t_signSGD.py and the rest of upstream
sys.path.insert(0, SQAT_ROOT)     # src/data.py, src/model_loader.py

from transformers import Trainer, TrainingArguments, set_seed  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from peft.tuners.lora.layer import CustomLoraLinear, IntLinear  # noqa: E402

from t_signSGD import tSignSGD  # noqa: E402

from src.data import build_data_collator, load_dataset_for_training  # noqa: E402
from src.model_loader import load_tokenizer  # noqa: E402

from lota_common import resolve_pretrained  # noqa: E402


class IntCustomTrainer(Trainer):
    """Upstream's IntCustomTrainer, with the sigma schedule fed the real step count."""

    def __init__(self, *args, lota_cfg=None, **kwargs):
        self.lota_cfg = lota_cfg
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        int_params = []
        for _, module in self.model.named_modules():
            if isinstance(module, IntLinear):
                int_params.extend(module.parameters())
        print(f"[LoTA] ternary adapter tensors: {len(int_params)}")
        if not int_params:
            raise RuntimeError(
                "no IntLinear modules found — the peft patch did not land, or "
                "lora_config._custom_modules did not match the quantized linears"
            )
        self.optimizer = tSignSGD(
            params=self.model.parameters(),
            int_params=int_params,
            threshold_ratio=self.lota_cfg["filter_ratio"],
            min_grad=self.lota_cfg["min_grad"],
            filter_upper=self.lota_cfg["filter_upper"],
        )
        return self.optimizer

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        # Upstream reads this out of a hardcoded per-dataset dict. The anneal is defined in
        # FRACTIONS of the run (0-80% / 80-100%), so a wrong total silently changes sigma_t.
        self.optimizer.create_scheduler(num_training_steps=num_training_steps)
        print(f"[LoTA] t-SignSGD sigma schedule over {num_training_steps} optimizer steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--quantized_model_dir", default=None, help="overrides lota.quantized_model_dir")
    ap.add_argument("--output_dir", default=None, help="overrides training.output_dir")
    # Smoke-test knobs. A 24h run that dies on step 3 costs a queue slot and a day, so the
    # wiring is exercised first with a handful of steps on a slice of the data.
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--max_train_samples", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    lota = cfg["lota"]
    if args.max_train_samples:
        cfg["data"]["max_train_samples"] = args.max_train_samples
    quantized_model_dir = args.quantized_model_dir or lota["quantized_model_dir"]
    output_dir = args.output_dir or cfg["training"]["output_dir"]
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(SQAT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    set_seed(cfg["training"]["seed"])

    print("=" * 70)
    print(f"  LoTA-QAF  |  base {quantized_model_dir}")
    print(f"  omega={lota['interval_point']}  sigma_t {lota['filter_ratio']} -> "
          f"{lota['min_grad']} -> {lota['filter_upper']}  residual={lota['residual']}")
    print(f"  out   {output_dir}")
    print("=" * 70)

    # --- model: GPTQ base + ternary adapters -------------------------------------------
    model = GPTQModel.load(
        quantized_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
        backend=BACKEND.AUTO_TRAINABLE,
    )
    model.optimize()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=list(cfg["lora"]["target_modules"]),
        bias=cfg["lora"].get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    lora_config._custom_modules = {BaseQuantLinear: CustomLoraLinear}
    lora_config.custom_config = {
        "residual": bool(lota["residual"]),
        "threshold": int(lota["interval_point"]),   # omega
    }
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- data: this repo's pipeline, byte-identical to every other row ------------------
    tokenizer = load_tokenizer(cfg, name=resolve_pretrained(cfg["model"]["name"]))
    os.chdir(SQAT_ROOT)  # data.train_dataset paths are repo-root relative
    train_dataset, _ = load_dataset_for_training(cfg, tokenizer)

    tcfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tcfg["num_epochs"],
        max_steps=args.max_steps,
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        group_by_length=tcfg.get("group_by_length", False),
        learning_rate=tcfg["learning_rate"],      # unused by t-SignSGD; kept for the record
        weight_decay=tcfg["weight_decay"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        max_grad_norm=None,                        # upstream disables clipping under LoTA-QAF
        bf16=tcfg.get("bf16", True),
        fp16=tcfg.get("fp16", False),
        logging_steps=tcfg["logging_steps"],
        save_strategy="no",                        # one final adapter; nothing to resume from
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        dataloader_num_workers=tcfg.get("dataloader_num_workers", 4),
        report_to=tcfg.get("report_to", "none"),
        run_name=os.path.basename(output_dir),
        seed=tcfg["seed"],
        label_names=["input_ids", "labels", "attention_mask"],  # required: see upstream
        remove_unused_columns=False,
    )

    trainer = IntCustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=build_data_collator(tokenizer),
        lota_cfg=lota,
    )
    model.config.use_cache = False
    torch.set_float32_matmul_precision("high")

    torch.cuda.reset_peak_memory_stats()
    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "lota_run.yaml"), "w") as f:
        yaml.safe_dump(
            {"lota": lota, "quantized_model_dir": os.path.abspath(quantized_model_dir),
             "config": os.path.abspath(args.config)},
            f,
        )
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"[LoTA] peak GPU memory: {peak:.0f} MB")
    print(f"[LoTA] adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
