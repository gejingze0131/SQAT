"""
Dataset loading and prompt formatting.
Supports: CommonsenseQA, Alpaca-style instruction data, MetaMath, custom datasets.
"""

import os
from typing import Optional, Tuple
from functools import partial

from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer


# ============================================================================
# Prompt Templates
# ============================================================================

COMMONSENSE_QA_TEMPLATE = (
    "Question: {question}\n"
    "Choices:\n"
    "A) {choice_A}\n"
    "B) {choice_B}\n"
    "C) {choice_C}\n"
    "D) {choice_D}\n"
    "E) {choice_E}\n"
    "Answer:"
)

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

ALPACA_INPUT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

METAMATH_TEMPLATE = (
    "Below is a math problem. Solve it step by step and put the final answer "
    "at the end.\n\n"
    "### Problem:\n{query}\n\n"
    "### Solution:\n"
)


# ============================================================================
# Formatting Functions
# ============================================================================

def _safe_strip(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _pick_first(example: dict, keys: list[str], default: str = "") -> str:
    for key in keys:
        if key in example:
            value = _safe_strip(example.get(key))
            if value:
                return value
    return default


def format_commonsense_qa(example: dict) -> dict:
    """Format a single CommonsenseQA example into prompt + completion."""
    choices = example["choices"]
    labels = choices["label"]
    texts = choices["text"]

    # Build a mapping: label -> text
    choice_map = dict(zip(labels, texts))

    prompt = COMMONSENSE_QA_TEMPLATE.format(
        question=example["question"],
        choice_A=choice_map.get("A", ""),
        choice_B=choice_map.get("B", ""),
        choice_C=choice_map.get("C", ""),
        choice_D=choice_map.get("D", ""),
        choice_E=choice_map.get("E", ""),
    )

    answer = example.get("answerKey", "")
    return {"text": prompt + " " + answer}


def format_alpaca(example: dict) -> dict:
    """Format Alpaca-style instruction data."""
    instruction = _pick_first(example, ["instruction", "prompt", "query"])
    input_text = _pick_first(example, ["input", "context"], default="")
    output = _pick_first(example, ["output", "response", "answer", "completion"], default="")

    if input_text:
        prompt = ALPACA_INPUT_TEMPLATE.format(
            instruction=instruction,
            input=input_text,
        )
    else:
        prompt = ALPACA_TEMPLATE.format(instruction=instruction)

    return {"text": prompt + output}


def format_metamath(example: dict) -> dict:
    query = _pick_first(
        example,
        [
            "query",
            "original_question",
            "problem",
            "question",
            "instruction",
            "prompt",
        ],
    )
    response = _pick_first(
        example,
        ["response", "solution", "output", "answer", "completion"],
    )

    if not query:
        skip_keys = {"type", "response", "solution", "output", "answer", "completion"}
        for k, v in example.items():
            if k in skip_keys:
                continue
            v = _safe_strip(v)
            if v:
                query = v
                break

    if not query:
        raise ValueError(f"Could not find a math prompt field in example keys: {list(example.keys())}")
    if not response:
        raise ValueError(f"Could not find a math response field in example keys: {list(example.keys())}")

    prompt = METAMATH_TEMPLATE.format(query=query)
    return {"text": prompt + response}

FORMATTERS = {
    "commonsense_qa": format_commonsense_qa,
    "alpaca": format_alpaca,
    "metamath": format_metamath,
    "metamathqa": format_metamath,
    "meta_math": format_metamath,
}


# ============================================================================
# Tokenization
# ============================================================================

def tokenize_fn(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
) -> dict:
    """Tokenize a batch of 'text' field examples for causal LM training."""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_len,
        padding=False,   # dynamic padding via data collator
    )
    # Labels = input_ids for causal LM (shifted internally by the model)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ============================================================================
# Helpers
# ============================================================================

def _load_raw_dataset(dataset_name: str):
    """
    Load dataset from HF hub or local path.

    Special handling:
      - commonsense_qa -> tau/commonsense_qa
      - alpaca         -> tatsu-lab/alpaca
      - metamath       -> meta-math/MetaMathQA
    """
    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower == "commonsense_qa":
        return load_dataset("tau/commonsense_qa")
    if dataset_name_lower == "alpaca":
        return load_dataset("tatsu-lab/alpaca")
    if dataset_name_lower in {"metamath", "metamathqa", "meta-math", "meta_math"}:
        return load_dataset("meta-math/MetaMathQA")

    if os.path.exists(dataset_name):
        # Custom local dataset (json/jsonl/csv/etc.)
        if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
            return load_dataset("json", data_files=dataset_name)
        if dataset_name.endswith(".csv"):
            return load_dataset("csv", data_files=dataset_name)
        # Fallback: try json loader first for local files
        return load_dataset("json", data_files=dataset_name)

    # Try loading from HF hub as provided
    return load_dataset(dataset_name)


def _resolve_formatter_name(data_cfg: dict, dataset_name: str) -> str:
    template_name = data_cfg.get("prompt_template")
    if template_name:
        return template_name.lower()

    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower in {"metamath", "metamathqa", "meta-math", "meta_math", "meta-math/metamathqa"}:
        return "metamath"

    return dataset_name_lower


def _resolve_split_name(raw, requested_split: Optional[str], fallback_candidates: list[str]) -> Optional[str]:
    """
    Resolve the split name more robustly.

    - If requested split exists, use it.
    - Else try fallbacks.
    - If dataset has only one split and request is train, use that split.
    """
    if requested_split and requested_split in raw:
        return requested_split

    for split in fallback_candidates:
        if split in raw:
            return split

    if requested_split == "train" and len(raw.keys()) == 1:
        return list(raw.keys())[0]

    return None


# ============================================================================
# Main Entry
# ============================================================================

def load_dataset_for_training(
    cfg: dict,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load and tokenize train/val datasets.

    Returns:
        train_dataset, eval_dataset (both tokenized HF Datasets)
    """
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    dataset_name = data_cfg["train_dataset"]
    max_seq_len = model_cfg["max_seq_len"]

    # --- Load raw dataset ---
    raw = _load_raw_dataset(dataset_name)

    # --- Format ---
    formatter_name = _resolve_formatter_name(data_cfg, dataset_name)
    formatter = FORMATTERS.get(formatter_name)
    if formatter is None:
        raise ValueError(
            f"Unknown prompt template '{formatter_name}'. "
            f"Available: {list(FORMATTERS.keys())}"
        )

    train_split_req = data_cfg.get("train_split", "train")
    val_split_req = data_cfg.get("val_split", "validation")

    train_split = _resolve_split_name(raw, train_split_req, ["train"])
    if train_split is None:
        raise ValueError(
            f"Could not resolve train split '{train_split_req}'. "
            f"Available splits: {list(raw.keys())}"
        )

    eval_split = None
    if val_split_req:
        eval_split = _resolve_split_name(raw, val_split_req, ["validation", "valid", "dev", "test"])

    train_raw = raw[train_split].map(
        formatter,
        remove_columns=raw[train_split].column_names,
    )

    eval_raw = None
    if eval_split is not None and eval_split != train_split:
        eval_raw = raw[eval_split].map(
            formatter,
            remove_columns=raw[eval_split].column_names,
        )

    # --- Optional train/val split from train set ---
    val_size = data_cfg.get("validation_size")
    if eval_raw is None and val_size:
        if isinstance(val_size, float):
            assert 0 < val_size < 1, "validation_size as float must be in (0, 1)"
        split_ds = train_raw.train_test_split(test_size=val_size, seed=cfg["training"]["seed"])
        train_raw = split_ds["train"]
        eval_raw = split_ds["test"]

    # --- Subsample ---
    max_train = data_cfg.get("max_train_samples")
    if max_train and max_train < len(train_raw):
        train_raw = train_raw.select(range(max_train))

    max_eval = data_cfg.get("max_eval_samples")
    if eval_raw is not None and max_eval and max_eval < len(eval_raw):
        eval_raw = eval_raw.select(range(max_eval))

    # --- Tokenize ---
    tok_fn = partial(tokenize_fn, tokenizer=tokenizer, max_seq_len=max_seq_len)

    train_dataset = train_raw.map(
        tok_fn,
        batched=True,
        remove_columns=["text"],
        num_proc=data_cfg.get("num_proc", 4),
        desc="Tokenizing train",
    )

    eval_dataset = None
    if eval_raw is not None:
        eval_dataset = eval_raw.map(
            tok_fn,
            batched=True,
            remove_columns=["text"],
            num_proc=data_cfg.get("num_proc", 4),
            desc="Tokenizing eval",
        )

    return train_dataset, eval_dataset


def load_calibration_data(
    cfg: dict,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    """
    Load a small calibration dataset for SQAT salient channel estimation.
    Uses the same training data, limited to sqat.calibration_samples.
    """
    sqat_cfg = cfg["qat"]["sqat"]
    n_samples = sqat_cfg["calibration_samples"]
    cal_seq_len = sqat_cfg["calibration_seq_len"]

    # Reuse training data
    data_cfg = cfg["data"]
    dataset_name = data_cfg["train_dataset"]
    raw = _load_raw_dataset(dataset_name)

    train_split = _resolve_split_name(raw, data_cfg.get("train_split", "train"), ["train"])
    if train_split is None:
        raise ValueError(
            f"Could not resolve calibration/train split for dataset '{dataset_name}'. "
            f"Available splits: {list(raw.keys())}"
        )

    raw_train = raw[train_split]
    if n_samples < len(raw_train):
        raw_train = raw_train.select(range(n_samples))

    formatter_name = _resolve_formatter_name(data_cfg, dataset_name)
    formatter = FORMATTERS[formatter_name]
    formatted = raw_train.map(formatter, remove_columns=raw_train.column_names)

    tok_fn = partial(tokenize_fn, tokenizer=tokenizer, max_seq_len=cal_seq_len)
    calibration = formatted.map(
        tok_fn,
        batched=True,
        remove_columns=["text"],
        num_proc=data_cfg.get("num_proc", 4),
        desc="Tokenizing calibration",
    )
    return calibration
