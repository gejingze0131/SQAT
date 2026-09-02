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
import hashlib
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


# Which span of each record carries the loss. data.loss_span in the yaml; "response" is the
# convention every result in results_saltq.csv before 2026-08-26 was produced with.
#
#   response              prompt masked, loss on the response only (PiSSA / LLM-Adapters' QLoRA
#                         lineage). On commonsense-170k the response is one of 15 fixed strings
#                         ("the correct answer is answer2"), so ~7 of its ~8 tokens are template
#                         and the whole task is ONE token per record. The training loss sits on
#                         a plateau at ~0.125 (= template learned, answer not) until an escape
#                         that at INT2 arrives late (0.35-0.94 epoch) or never.
#   instruction+response  only the fixed PROMPT header ("Below is an instruction ... ###
#                         Instruction:\n") is masked; the question + options + response are
#                         supervised. 100-300 real language-modelling tokens per record, which
#                         is what MetaMath's rationales give math for free -- and math has no
#                         plateau, no instability and zp_lr 3x higher without blowing up.
#                         LLM-Adapters' own train_on_inputs=True default is this mode.
#   full                  everything, header included.
# A different loss_span is a different EXPERIMENTAL CELL: every baseline it is compared with
# must be re-run under the same span.
LOSS_SPANS = ("response", "instruction+response", "full")


def _header_len(tokenizer: PreTrainedTokenizer) -> int:
    """Token count of PROMPT's fixed prefix, up to and including "### Instruction:\n"."""
    header = PROMPT.split("{instruction}")[0]
    return len(tokenizer(header, truncation=False).input_ids)


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizer,
    loss_span: str = "response",
) -> Dict:
    """Tokenize source+target and mask the unsupervised span out of the labels."""
    if loss_span not in LOSS_SPANS:
        raise ValueError(f"data.loss_span must be one of {LOSS_SPANS}, got {loss_span!r}")
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    header_len = _header_len(tokenizer) if loss_span == "instruction+response" else 0
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        if loss_span == "response":
            masked = source_len
        elif loss_span == "instruction+response":
            # The header is tokenized on its own; BPE could merge differently at its last
            # token when the instruction follows, so never mask past the prompt itself.
            masked = min(header_len, source_len)
        else:
            masked = 0
        label[:masked] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def train_tokenize_function(examples, tokenizer, query: str, response: str,
                            loss_span: str = "response") -> Dict:
    """Batched `datasets.map` function: {query, response} columns -> {input_ids, labels}."""
    sources = [PROMPT.format_map(dict(instruction=q)) for q in examples[query]]
    targets = [f"{r}\n{tokenizer.eos_token}" for r in examples[response]]
    return preprocess(sources, targets, tokenizer, loss_span=loss_span)


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
# Raw-text causal LM (WikiText-2) — the GPTQ-lineage protocol
# ============================================================================
#
# The instruction pipeline above renders every record through PROMPT and supervises a response
# span. WikiText-2 has no instruction and no response: the task IS language modelling, and the
# metric is perplexity. Mixing the two would be a silent error in both directions — a PROMPT
# header prepended to every 1024-token block of Wikipedia, and a perplexity that is not
# comparable to any published number.
#
# So `data.task_type: lm` switches to the convention the GPTQ / AWQ / OmniQuant / LoftQ / ApiQ
# line all use, and which the WikiText-2 numbers in the literature are produced with:
#
#   1. JOIN every row's raw text with "\n\n" into ONE string (rows of wikitext-2-raw-v1 are
#      lines, most of them fragments, many empty — tokenizing them individually and inserting an
#      EOS per row, which is what `calibration_source` does for C4 documents, would put ~36k
#      spurious EOS into a 2.4M-token stream).
#   2. Tokenize that string ONCE. add_special_tokens is left at its default, so exactly one BOS
#      sits at the front of the whole stream — again the lineage's convention, not a choice.
#   3. Cut the token stream into NON-OVERLAPPING windows of `block_size`. The tail that does not
#      fill a window is dropped (torch's `nsamples = numel // seqlen`).
#   4. labels == input_ids: EVERY token carries loss. There is no mask here.
#
# Training and evaluation share steps 1-3 through `lm_token_stream`, which is the point: the
# blocks a model is fine-tuned on and the windows its perplexity is measured over are cut by the
# same code from the same kind of stream, so `block_size` is one number, not two that can drift.
#
# `wikitext-2-raw-v1` vs `wikitext-2-v1` is NOT a detail: the processed variant replaces rare
# words with <unk>, which lowers perplexity by a wide margin. Perplexities across the two are not
# comparable. datasets/wikitext2 is the RAW variant.

LM_TEXT_JOIN = "\n\n"


def _lm_raw_texts(data_path: str, split: str, text_field: str = "text") -> List[str]:
    """Every row's raw text for one split, in file order.

    A bare JSON FILE is accepted as well as a `<dir>/<split>.json` dataset, and its top level may
    be a list of strings — which `datasets` cannot load, and which is the shape of the C4 control
    files (datasets/c4_calib_1024.json, datasets/c4_val_ppl.json). `split` is ignored for a file.
    """
    if os.path.isfile(data_path):
        import json as _json
        with open(data_path, encoding="utf-8") as f:
            raw = _json.load(f)
        return [r if isinstance(r, str) else r[text_field] for r in raw]

    ds = _load_raw_split(data_path, split)
    if ds is None:
        raise FileNotFoundError(
            f"Could not load split '{split}' from '{data_path}' for a task_type=lm dataset."
        )
    if text_field not in ds.column_names:
        raise ValueError(
            f"data.text_field='{text_field}' is not a column of {data_path}/{split} "
            f"(columns: {ds.column_names})."
        )
    return list(ds[text_field])


def lm_token_stream(
    data_path: str,
    split: str,
    tokenizer: PreTrainedTokenizer,
    text_field: str = "text",
    join: str = LM_TEXT_JOIN,
) -> np.ndarray:
    """Steps 1-2 above: one split -> one int64 token stream.

    Cached on disk under HF_HOME, keyed by everything that can change the result. Without it the
    11 MB WikiText-2 train string is re-tokenized once per DDP rank on every launch, and again by
    the offline calibration stage.
    """
    key = "|".join([
        os.path.abspath(data_path), split, text_field, join,
        getattr(tokenizer, "name_or_path", ""),
        str(getattr(tokenizer, "vocab_size", "")),
    ])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    cache_root = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "sqat_lm_streams")
    cache_path = os.path.join(cache_root, f"{digest}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    text = join.join(_lm_raw_texts(data_path, split, text_field))
    prev_max_len = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e12)          # the whole split is ONE sequence here
    try:
        ids = tokenizer(text, truncation=False).input_ids
    finally:
        tokenizer.model_max_length = prev_max_len
    stream = np.asarray(ids, dtype=np.int64)

    os.makedirs(cache_root, exist_ok=True)
    # np.save APPENDS ".npy" unless the name already ends with it, so the temp file is written
    # through a handle: a ".tmp" path would land at ".tmp.npy" and the rename would miss it.
    tmp = f"{cache_path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, stream)
    os.replace(tmp, cache_path)
    return stream


def lm_windows(stream: np.ndarray, seq_len: int) -> np.ndarray:
    """Step 3: [n_windows, seq_len] of NON-OVERLAPPING windows; the ragged tail is dropped."""
    n = stream.shape[0] // seq_len
    if n == 0:
        raise ValueError(f"token stream of {stream.shape[0]} tokens is shorter than one "
                         f"{seq_len}-token window")
    return stream[: n * seq_len].reshape(n, seq_len)


def _lm_dataset(windows: np.ndarray) -> Dataset:
    """Step 4: every token supervised."""
    rows = [w for w in windows]
    return Dataset.from_dict({"input_ids": rows, "labels": [w.copy() for w in rows]})


def load_lm_blocks(cfg: dict, tokenizer: PreTrainedTokenizer, split: str) -> Dataset:
    """One split of a `task_type: lm` dataset as fixed-length, fully supervised blocks."""
    data_cfg = cfg["data"]
    block_size = int(data_cfg.get("block_size") or cfg["model"]["max_seq_len"])
    stream = lm_token_stream(
        data_cfg["train_dataset"], split, tokenizer,
        text_field=str(data_cfg.get("text_field", "text")),
        join=str(data_cfg.get("text_join", LM_TEXT_JOIN)),
    )
    windows = lm_windows(stream, block_size)
    print(f"[Data] lm {data_cfg['train_dataset']}/{split}: {stream.shape[0]} tokens -> "
          f"{windows.shape[0]} x {block_size} blocks (every token supervised)")
    return _lm_dataset(windows)


def _load_lm_for_training(cfg: dict, tokenizer: PreTrainedTokenizer):
    """`load_dataset_for_training`'s task_type=lm branch."""
    data_cfg = cfg["data"]
    train = load_lm_blocks(cfg, tokenizer, data_cfg.get("train_split", "train"))
    if data_cfg.get("shuffle_dataset", True):
        train = train.shuffle(seed=cfg["training"]["seed"])
    max_train = data_cfg.get("max_train_samples")
    if max_train and max_train < len(train):
        train = train.select(range(max_train))

    eval_ds = None
    val_split = data_cfg.get("val_split")
    if val_split:
        eval_ds = load_lm_blocks(cfg, tokenizer, val_split)
        max_eval = data_cfg.get("max_eval_samples")
        if max_eval and max_eval < len(eval_ds):
            eval_ds = eval_ds.select(range(max_eval))
    return train, eval_ds


def _load_lm_calibration(cfg: dict, tokenizer: PreTrainedTokenizer) -> Dataset:
    """`load_calibration_data`'s task_type=lm branch: the task's own text, per the brief.

    The calibration budget is stated in TOKENS (calibration_samples x calibration_seq_len)
    because that is the quantity the GPTQ Hessian actually sees, and it is the axis the
    calibration ablation moves. Windows are drawn from the SAME non-overlapping cut the training
    blocks come from, shuffled with the run seed and truncated — so the set is reproducible from
    (seed, samples, seq_len) alone and never overlaps itself.
    """
    sqat_cfg = cfg["qat"]["sqat"]
    n_samples = int(sqat_cfg["calibration_samples"])
    seq_len = int(sqat_cfg["calibration_seq_len"])
    data_cfg = cfg["data"]
    stream = lm_token_stream(
        data_cfg["train_dataset"], data_cfg.get("train_split", "train"), tokenizer,
        text_field=str(data_cfg.get("text_field", "text")),
        join=str(data_cfg.get("text_join", LM_TEXT_JOIN)),
    )
    windows = lm_windows(stream, seq_len)
    if n_samples > windows.shape[0]:
        raise ValueError(
            f"calibration_samples={n_samples} exceeds the {windows.shape[0]} non-overlapping "
            f"{seq_len}-token windows in {data_cfg['train_dataset']}/"
            f"{data_cfg.get('train_split', 'train')}")
    rng = np.random.default_rng(int(cfg.get("training", {}).get("seed", 42)))
    pick = rng.permutation(windows.shape[0])[:n_samples]
    pick.sort()
    print(f"[Data] calibration: {n_samples} in-domain LM windows x {seq_len} tokens "
          f"({n_samples * seq_len} tokens) from {data_cfg['train_dataset']}/"
          f"{data_cfg.get('train_split', 'train')}, seeded draw from "
          f"{windows.shape[0]} non-overlapping windows")
    return _lm_dataset(windows[pick])


def is_lm_task(cfg: dict) -> bool:
    """True when data.task_type selects the raw-text causal-LM pipeline."""
    return str(cfg.get("data", {}).get("task_type", "instruction")).lower() == "lm"


# ============================================================================
# Main entry
# ============================================================================

def _tokenize(ds: Dataset, tokenizer, fields: Sequence[str], num_proc: int,
              desc: str, loss_span: str = "response") -> Dataset:
    # loss_span rides in fn_kwargs, which datasets folds into the map fingerprint -- so a
    # different span gets its own cache file instead of silently reusing "response" labels.
    return ds.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        load_from_cache_file=True,
        desc=desc,
        fn_kwargs={"tokenizer": tokenizer, "query": fields[0], "response": fields[1],
                   "loss_span": loss_span},
    )


def supervised_token_share(ds: Dataset, n: int = 2000) -> Tuple[float, float]:
    """(mean supervised tokens per record, share of all tokens) over the first n records."""
    sup = tot = 0
    for rec in ds.select(range(min(n, len(ds)))):
        lab = np.asarray(rec["labels"])
        sup += int((lab != IGNORE_INDEX).sum()); tot += lab.size
    n = min(n, len(ds))
    return sup / max(n, 1), sup / max(tot, 1)


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

    # data.task_type: lm -> raw text, no PROMPT, no mask, fixed-length blocks (see above).
    if is_lm_task(cfg):
        return _load_lm_for_training(cfg, tokenizer)

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

    loss_span = str(data_cfg.get("loss_span", "response"))
    train_dataset = _tokenize(train_raw, tokenizer, fields, num_proc, "Tokenizing train",
                              loss_span=loss_span)
    eval_dataset = None
    if eval_raw is not None:
        eval_dataset = _tokenize(eval_raw, tokenizer, fields, num_proc, "Tokenizing eval",
                                 loss_span=loss_span)
    # Visible evidence of which span is actually supervised (a stale tokenizer cache would
    # otherwise be indistinguishable from the intended mode).
    per_rec, share = supervised_token_share(train_dataset)
    print(f"[Data] loss_span={loss_span}: {per_rec:.1f} supervised tokens/record "
          f"({share:.1%} of all tokens, first 2000 records)")

    return train_dataset, eval_dataset


def _generic_text_windows(path: str, tokenizer: PreTrainedTokenizer, n_windows: int,
                          seq_len: int) -> Dataset:
    """calibration_source: raw texts -> EOS-joined token stream -> n_windows x seq_len rows."""
    import json as _json
    texts = _json.load(open(path))
    texts = [t["text"] if isinstance(t, dict) else t for t in texts]
    eos = tokenizer.eos_token_id
    ids: List[int] = []
    for t in texts:
        ids.extend(tokenizer(t, add_special_tokens=False).input_ids + [eos])
        if len(ids) >= n_windows * seq_len:
            break
    n_avail = len(ids) // seq_len
    if n_avail < n_windows:
        raise ValueError(f"{path} yields {n_avail} windows of {seq_len} tokens, fewer than the "
                         f"{n_windows} requested")
    rows = [ids[i * seq_len:(i + 1) * seq_len] for i in range(n_windows)]
    print(f"[Data] calibration: {n_windows} generic-text windows x {seq_len} tokens from {path} "
          f"({n_windows * seq_len} tokens)")
    return Dataset.from_dict({"input_ids": [np.array(r) for r in rows],
                              "labels": [np.array(r) for r in rows]})


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
    n_samples = int(sqat_cfg["calibration_samples"])
    cal_seq_len = sqat_cfg["calibration_seq_len"]
    sampling = str(sqat_cfg.get("calibration_sampling", "first"))
    source = sqat_cfg.get("calibration_source")
    # A task_type=lm dataset calibrates on its OWN text, cut the same way its training blocks
    # are. calibration_source still wins when set, because "generic C4 text vs the task's own
    # text" is exactly the ablation this switch exists for.
    if source is None and is_lm_task(cfg):
        return _load_lm_calibration(cfg, tokenizer)
    if source:
        # GENERIC-TEXT calibration (the GPTQ-paper recipe): a JSON list of raw strings (e.g.
        # datasets/c4_calib_1024.json), concatenated with EOS and cut into calibration_seq_len
        # token windows; calibration_samples windows are kept. No prompt template, no labels.
        # 128 x 2048 windows = 262k tokens = the standard budget. Out-of-domain by design.
        return _generic_text_windows(source, tokenizer, n_samples, int(cal_seq_len))

    raw, fields = _prepare_raw(cfg, cfg["data"].get("train_split", "train"))
    if raw is None:
        raise ValueError(f"Could not load calibration data from '{cfg['data']['train_dataset']}'.")
    # HOW THE RECORDS ARE PICKED. "first" is what every base before 2026-08-28 used: the first n
    # records of the raw file. datasets/commonsense/train.json is grouped by task (boolq is
    # records 0..9426), so "first" calibrated every Hessian AND every saliency statistic in the
    # project on boolq prompts only -- 128 records x ~74 tokens = 9.5k tokens of one template.
    # Kept as the default so those bases stay reproducible. "shuffle" draws n records uniformly
    # (seeded); "balanced" draws n / #types per `type` (seeded).
    seed = int(cfg.get("training", {}).get("seed", 42))
    if sampling == "first":
        if n_samples < len(raw):
            raw = raw.select(range(n_samples))
    elif sampling == "shuffle":
        raw = raw.shuffle(seed=seed)
        if n_samples < len(raw):
            raw = raw.select(range(n_samples))
    elif sampling == "balanced":
        if "type" not in raw.column_names:
            raise ValueError("calibration_sampling=balanced needs a `type` column in the raw data")
        types = sorted(set(raw["type"]))
        per = -(-n_samples // len(types))                       # ceil
        parts = []
        for i, t in enumerate(types):
            sub = raw.filter(lambda r, t=t: r["type"] == t).shuffle(seed=seed + i)
            parts.append(sub.select(range(min(per, len(sub)))))
        raw = concatenate_datasets(parts).shuffle(seed=seed)
        if n_samples < len(raw):
            raw = raw.select(range(n_samples))
    else:
        raise ValueError(f"qat.sqat.calibration_sampling must be first|shuffle|balanced, got {sampling!r}")
    if "type" in raw.column_names:
        import collections
        print(f"[Data] calibration: {len(raw)} records, sampling={sampling}, "
              f"task mix={dict(collections.Counter(raw['type']))}")

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
