"""
Dataset loading, prompt formatting and collation.

One pipeline, following the PiSSA / LLM-Adapters convention (see migration/train.py):

    raw record -> {instruction, output} -> PROMPT + response -> loss on the RESPONSE ONLY

Two things here differ from a plain causal-LM `text` pipeline and both matter for the
scores this repo reports:

  * the prompt tokens are masked out of the loss (IGNORE_INDEX). Training on the prompt
    as well spends capacity — which at INT2/INT3 is exactly what is scarce — on
    reproducing a fixed template.
  * the response is terminated with an explicit EOS. Without it the model never learns to
    stop, and a generative evaluator (scripts/gen_vllm.py) reads whatever it keeps
    emitting past the answer.

Datasets live under datasets/<name>/{train,test}.json as a flat JSON array of records with
the fields `instruction` / `input` / `output` / `type` (the pissa-dataset schema). `input`
is empty for every record of both commonsense and metamath, and is ignored here exactly as
migration/train.py ignores it; a non-empty one raises rather than being silently dropped.
"""

import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizer

IGNORE_INDEX = -100

# The prompt every training record is rendered through. Kept byte-identical to
# migration/train.py — the test splits ship their `instruction` field ALREADY wrapped in this
# exact string, so any drift here silently creates a train/test prompt mismatch.
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

DEFAULT_DATASET_FIELD = ["instruction", "output"]


# ============================================================================
# Record adapters — normalize a raw dataset row to {instruction, output}
# ============================================================================

def adapt_commonsense_qa(example: dict) -> dict:
    """tau/commonsense_qa -> the instruction/output schema.

    That hub dataset stores the choices as a parallel {label: [...], text: [...]} struct and
    the gold answer as a bare letter, so it cannot be read through `dataset_field` like the
    local JSON datasets can.
    """
    choices = example["choices"]
    rendered = "\n".join(
        f"{label}) {text}" for label, text in zip(choices["label"], choices["text"])
    )
    instruction = (
        f"Please answer the following multiple-choice question with the letter of the "
        f"correct option.\n\nQuestion: {example['question']}\n{rendered}\n\n"
        f"Answer format: A/B/C/D/E"
    )
    return {"instruction": instruction, "output": example.get("answerKey", "")}


# Only datasets whose raw schema is NOT already (instruction, output) need an entry.
RECORD_ADAPTERS = {
    "commonsense_qa": adapt_commonsense_qa,
}


# ============================================================================
# Tokenization — prompt masked, response supervised
# ============================================================================

def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(text, max_length=tokenizer.model_max_length, truncation=True)
        for text in strings
    ]
    input_ids = labels = [np.array(tok.input_ids) for tok in tokenized_list]
    input_ids_lens = labels_lens = [len(tok.input_ids) for tok in tokenized_list]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    """Tokenize source+target and mask the source span out of the labels."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def train_tokenize_function(examples, tokenizer, query: str, response: str) -> Dict:
    """Batched `datasets.map` function: {query, response} columns -> {input_ids, labels}."""
    sources = [PROMPT.format_map(dict(instruction=q)) for q in examples[query]]
    targets = [f"{r}\n{tokenizer.eos_token}" for r in examples[response]]
    return preprocess(sources, targets, tokenizer)


# ============================================================================
# Collation
# ============================================================================

@dataclass
class DataCollatorForSupervisedDataset:
    """Right-pad a batch: input_ids with pad_token_id, labels with IGNORE_INDEX."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = [torch.as_tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.as_tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def build_data_collator(tokenizer: PreTrainedTokenizer) -> DataCollatorForSupervisedDataset:
    """The single collator used by training, evaluation and QAT calibration alike."""
    return DataCollatorForSupervisedDataset(tokenizer=tokenizer)


# ============================================================================
# Raw loading
# ============================================================================

def _split_files(data_path: str, split: str) -> Optional[str]:
    """Resolve <data_path> + <split> to a local json file, or None if it is not local."""
    if os.path.isfile(data_path):
        return data_path
    if os.path.isdir(data_path):
        for ext in (".json", ".jsonl"):
            candidate = os.path.join(data_path, f"{split}{ext}")
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"No '{split}.json' / '{split}.jsonl' under {data_path}. "
            f"Found: {sorted(os.listdir(data_path))}"
        )
    return None


def _load_raw_split(data_path: str, split: str) -> Optional[Dataset]:
    """Load one split from a local json file/dir or from the HF hub. None if absent."""
    local = _split_files(data_path, split)
    if local is not None:
        return load_dataset("json", data_files=local, split="train")

    try:
        return load_dataset(data_path, split=split)
    except (ValueError, KeyError):
        return None


def _select_sub_tasks(ds: Dataset, sub_task: Sequence[str], rank: int = 0) -> Dataset:
    """Keep only the requested `type`s, honouring the `name:N` cap of migration/train.py.

    The datasets here are single flat files with a `type` column (boolq / gsm8k / ...), so a
    sub-task is a filter rather than a separate file. `gsm8k:500` takes the first 500 records
    of that type.
    """
    if "type" not in ds.column_names:
        raise ValueError(
            f"sub_task={list(sub_task)} requested but the dataset has no 'type' column "
            f"(columns: {ds.column_names})."
        )

    parts = []
    for task in sub_task:
        name, _, cap = task.partition(":")
        part = ds.filter(lambda ex, n=name: ex["type"] == n)
        if len(part) == 0:
            raise ValueError(
                f"sub_task '{name}' matched 0 records. Available types: "
                f"{sorted(set(ds['type']))}"
            )
        if cap:
            part = part.select(range(min(int(cap), len(part))))
        if rank == 0:
            print(f"[Data]   sub_task {name}: {len(part)} records")
        parts.append(part)
    return concatenate_datasets(parts)


def _to_instruction_output(ds: Dataset, cfg_template: Optional[str], data_path: str,
                           dataset_field: Sequence[str]) -> Tuple[Dataset, List[str]]:
    """Apply a record adapter if the raw schema is not already (instruction, output)."""
    key = (cfg_template or os.path.basename(os.path.normpath(data_path))).lower()
    adapter = RECORD_ADAPTERS.get(key)
    if adapter is None:
        missing = [f for f in dataset_field if f not in ds.column_names]
        if missing:
            raise ValueError(
                f"Dataset '{data_path}' is missing the field(s) {missing} required by "
                f"data.dataset_field={list(dataset_field)}. Columns: {ds.column_names}. "
                f"Register a record adapter in src/data.RECORD_ADAPTERS if its schema differs."
            )
        if "input" in ds.column_names:
            non_empty = sum(1 for v in ds["input"] if str(v).strip())
            if non_empty:
                raise ValueError(
                    f"{non_empty} records of '{data_path}' carry a non-empty `input` field. "
                    f"PROMPT has no slot for it, so training would silently drop that context."
                )
        return ds, list(dataset_field)

    adapted = ds.map(adapter, remove_columns=ds.column_names, desc="Adapting records")
    return adapted, list(DEFAULT_DATASET_FIELD)


# ============================================================================
# Main entry
# ============================================================================

def _tokenize(ds: Dataset, tokenizer, fields: Sequence[str], num_proc: int,
              desc: str) -> Dataset:
    return ds.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        load_from_cache_file=True,
        desc=desc,
        fn_kwargs={"tokenizer": tokenizer, "query": fields[0], "response": fields[1]},
    )


def _prepare_raw(cfg: dict, split: str, rank: int = 0) -> Tuple[Optional[Dataset], List[str]]:
    """Load one split and normalize it to (instruction, output). None if the split is absent."""
    data_cfg = cfg["data"]
    data_path = data_cfg["train_dataset"]
    ds = _load_raw_split(data_path, split)
    if ds is None:
        return None, list(data_cfg.get("dataset_field") or DEFAULT_DATASET_FIELD)

    sub_task = data_cfg.get("sub_task")
    if sub_task:
        ds = _select_sub_tasks(ds, sub_task, rank=rank)

    ds, fields = _to_instruction_output(
        ds,
        data_cfg.get("prompt_template"),
        data_path,
        data_cfg.get("dataset_field") or DEFAULT_DATASET_FIELD,
    )
    return ds, fields


def load_dataset_for_training(
    cfg: dict,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Load and tokenize the train/eval splits.

    Returns (train_dataset, eval_dataset); eval_dataset is None unless data.val_split or
    data.validation_size asks for one.
    """
    data_cfg = cfg["data"]
    num_proc = int(data_cfg.get("num_proc", 32))

    train_raw, fields = _prepare_raw(cfg, data_cfg.get("train_split", "train"))
    if train_raw is None:
        raise ValueError(
            f"Could not load train split '{data_cfg.get('train_split', 'train')}' from "
            f"'{data_cfg['train_dataset']}'."
        )

    print(f"[Data] {data_cfg['train_dataset']} train: {len(train_raw)} records")
    for k, v in train_raw[0].items():
        print(f"[Data]   {k}: {str(v)[:200]}")

    if data_cfg.get("shuffle_dataset", True):
        train_raw = train_raw.shuffle(seed=cfg["training"]["seed"])

    eval_raw = None
    val_split = data_cfg.get("val_split")
    if val_split:
        eval_raw, _ = _prepare_raw(cfg, val_split)

    val_size = data_cfg.get("validation_size")
    if eval_raw is None and val_size:
        split_ds = train_raw.train_test_split(
            test_size=val_size, seed=cfg["training"]["seed"]
        )
        train_raw, eval_raw = split_ds["train"], split_ds["test"]

    max_train = data_cfg.get("max_train_samples")
    if max_train and max_train < len(train_raw):
        train_raw = train_raw.select(range(max_train))

    max_eval = data_cfg.get("max_eval_samples")
    if eval_raw is not None and max_eval and max_eval < len(eval_raw):
        eval_raw = eval_raw.select(range(max_eval))

    train_dataset = _tokenize(train_raw, tokenizer, fields, num_proc, "Tokenizing train")
    eval_dataset = None
    if eval_raw is not None:
        eval_dataset = _tokenize(eval_raw, tokenizer, fields, num_proc, "Tokenizing eval")

    return train_dataset, eval_dataset


def load_calibration_data(
    cfg: dict,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    """A small slice of the TRAINING data, tokenized identically, for SQAT/SALT-Q calibration.

    Salient-channel statistics have to be estimated on the activation distribution the model
    will actually see, so this deliberately shares PROMPT, masking and truncation with
    load_dataset_for_training; only the sequence-length cap differs.
    """
    sqat_cfg = cfg["qat"]["sqat"]
    n_samples = sqat_cfg["calibration_samples"]
    cal_seq_len = sqat_cfg["calibration_seq_len"]

    raw, fields = _prepare_raw(cfg, cfg["data"].get("train_split", "train"))
    if raw is None:
        raise ValueError(f"Could not load calibration data from '{cfg['data']['train_dataset']}'.")
    if n_samples < len(raw):
        raw = raw.select(range(n_samples))

    # model_max_length is what _tokenize_fn truncates against; calibration uses its own cap.
    prev_max_len = tokenizer.model_max_length
    tokenizer.model_max_length = cal_seq_len
    try:
        return _tokenize(
            raw, tokenizer, fields,
            num_proc=min(int(cfg["data"].get("num_proc", 32)), 8),
            desc="Tokenizing calibration",
        )
    finally:
        tokenizer.model_max_length = prev_max_len
