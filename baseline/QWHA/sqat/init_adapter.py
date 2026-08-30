"""
QWHA adapter initialization (AdaAlloc parameter selection + value refinement) for one cell.

Upstream's `src/init/initialize.py` in three stages -- quantization error, a calibration pass
that accumulates X^T X per projection, then the per-layer spectrum selection -- with
`compute_quant_error` and `initialize_adapter` called from upstream unchanged. Two things differ:

  1. THE CALIBRATION SET. Upstream uses wikitext2. This repo measured that generic text is what
     breaks low-bit calibration here: the same merged checkpoint scored 36.64 (INT2) on a
     first-N BoolQ-only set and 66.22 on a task-balanced 3500-record in-domain set, while
     standard C4 128x2048 loses the instruction template outright (30.67). The QWHA base is
     built on the balanced set (make_bcal_base.py), so the X^T X that decides WHERE the spectrum
     goes is estimated on the same records -- `src/data.load_calibration_data`, prompt template
     included, padding masked out of the accumulation. `--calib wikitext2` keeps upstream's
     path available for provenance.
  2. MEMORY. Upstream runs the eigendecompositions while the fp32 base is still resident (it had
     80 GB). Here the pass and the roots are separate steps with the base moved back to the host
     in between, and the quantized model leaves the GPU for the duration. Same damping, same
     clamp, same R.
"""

import argparse
import gc
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# qwha_common first: importing it is what puts upstream's src/ and src/init/ on sys.path.
from qwha_common import (SQAT_REPO, build_qwha_model, gptq_base_dir, init_ckpt_dir,
                         sqat_data_module)
import initialize as up          # upstream src/init/initialize.py


# --------------------------------------------------------------------------- calibration

def register_masked_hooks(model, state):
    """Upstream `register_hook`, with padding excluded from the accumulation.

    Upstream's calibration is dense 4096-token wikitext2 windows, so every position carries
    signal. This repo's calibration records are variable-length instruction records that the
    collator right-pads; accumulating X^T X over the pad positions would put a fixed token's
    outer product into every Hessian.
    """
    def generate_hook(layer_name: str):
        def accumulate_xtx_hook(layer, inputs):
            x = inputs[0]
            M = x.shape[-1]
            device = layer.weight.device
            x = x.reshape(-1, M).to(device)
            mask = state.get("mask")
            if mask is not None:
                x = x[mask.reshape(-1)]
            if x.numel() == 0:
                return
            if not hasattr(layer, "xtx_buffer"):
                layer.register_buffer("xtx_buffer", torch.zeros((M, M), dtype=torch.float64))
            xtx = (x.T @ x).cpu()
            if not xtx.isnan().any() and not xtx.isinf().any():
                layer.xtx_buffer += xtx
            else:                                  # retry in float64, as upstream does
                x = x.to(torch.float64)
                layer.xtx_buffer += (x.T @ x).cpu()

        return accumulate_xtx_hook

    model.hook_handles = []
    for name, module in model.named_modules():
        if name[-4:] == "proj":
            model.hook_handles.append(module.register_forward_pre_hook(generate_hook(name)))


def calibration_pass_bcal(model, cfg, tokenizer, batch_size: int):
    """X^T X over this repo's balanced in-domain calibration records."""
    data = sqat_data_module()
    cal = data.load_calibration_data(cfg, tokenizer)
    dl = DataLoader(cal, batch_size=batch_size, shuffle=False,
                    collate_fn=data.build_data_collator(tokenizer))
    state = {}
    register_masked_hooks(model, state)
    for batch in tqdm(dl, "Calibration forward (balanced, in-domain)"):
        input_ids = batch["input_ids"].to(model.device)
        attn = batch["attention_mask"].to(model.device)
        state["mask"] = attn.bool()
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attn)
    for handle in model.hook_handles:
        handle.remove()
    del model.hook_handles


def calibration_pass_wikitext2(model, tokenizer, batch_size: int = 1):
    """Upstream `calibration_forward`, minus its trailing call to the root computation."""
    from datasets import load_dataset

    max_length = min(model.config.max_position_embeddings, 4096)
    up.register_hook(model)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    seq_len = encodings.input_ids.size(1)
    for begin_loc in tqdm(range(0, seq_len, max_length * batch_size), "Calibration forward"):
        end_loc = min(begin_loc + max_length * batch_size, seq_len)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        if input_ids.shape[1] < max_length * batch_size:
            num_append = max_length * batch_size - input_ids.shape[1]
            input_ids = torch.cat(
                (input_ids, torch.IntTensor([[0] * num_append]).to(input_ids.device)), dim=1)
            target_ids = torch.cat(
                (target_ids, torch.IntTensor([[-100] * num_append]).to(input_ids.device)), dim=1)
        input_ids = input_ids.reshape(batch_size, max_length)
        target_ids = target_ids.reshape(batch_size, max_length)
        with torch.inference_mode():
            model(input_ids, labels=target_ids)

    for handle in model.hook_handles:
        handle.remove()
    del model.hook_handles


def compute_roots(model, device="cuda"):
    """Upstream `clear_hook_and_calculate_root`'s arithmetic, on an explicit device."""
    modules = [m for _, m in model.named_modules() if hasattr(m, "xtx_buffer")]
    for module in tqdm(modules, "Calculating matrix root"):
        H = module.xtx_buffer.to(device)
        avg_diag_H = H[range(H.shape[0]), range(H.shape[0])].mean()
        H += torch.diag(torch.Tensor([0.0001 * avg_diag_H] * H.shape[0])).to(H.device)
        eigenval, eigenvec = torch.linalg.eigh(H)
        sqrt_eigenval = torch.sqrt(torch.clamp(eigenval, min=0))
        R = torch.diag(sqrt_eigenval) @ eigenvec.T
        module.register_buffer("xtx_root", R.cpu())
        del module.xtx_buffer, H, eigenval, eigenvec, R
        gc.collect()
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="SQAT-style QWHA config; supplies the balanced in-domain calibration "
                         "set, the base directory and the rank/scale. Required unless "
                         "--calib wikitext2 is used with explicit -m/-b/-g/-r.")
    ap.add_argument("--gptq_dir", default=None, help="override the quantized base directory")
    ap.add_argument("--calib", default="balanced", choices=["balanced", "wikitext2"],
                    help="which records the X^T X is estimated on")
    ap.add_argument("-m", "--model_id", default=None)
    ap.add_argument("-b", "--bits", type=int, default=None)
    ap.add_argument("-g", "--group_size", type=int, default=None)
    ap.add_argument("-r", "--rank", type=int, default=None)
    ap.add_argument("-s", "--scale", type=float, default=0.25,
                    help="scaling the spectrum is STORED at; training replays it at its own "
                         "scale (upstream's initialize.py default, kept)")
    ap.add_argument("-a", "--alpha", type=float, default=1.0, help="AdaAlloc exponent (upstream 1.0)")
    ap.add_argument("--calib_batch_size", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = None
    if args.config:
        os.chdir(SQAT_REPO)          # data.train_dataset paths are repo-root relative
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model_id = args.model_id or cfg["model"]["name"]
        bits = args.bits or int(cfg["model"]["quant_bits"])
        group_size = args.group_size or int(cfg["qwha"]["group_size"])
        rank = args.rank or int(cfg["qwha"]["rank"])
        gptq_path = args.gptq_dir or cfg["qwha"].get("gptq_base_dir") \
            or gptq_base_dir(model_id, bits, group_size)
    else:
        if args.calib != "wikitext2":
            raise SystemExit("--config is required unless --calib wikitext2")
        model_id = args.model_id or "meta-llama/Llama-2-7b-hf"
        bits, group_size, rank = args.bits, args.group_size, args.rank or 64
        if bits is None or group_size is None:
            raise SystemExit("-b/--bits and -g/--group_size are required without --config")
        gptq_path = args.gptq_dir or gptq_base_dir(model_id, bits, group_size)

    up.ALPHA = args.alpha
    out = args.out or (cfg["qwha"].get("init_ckpt_dir") if cfg else None) \
        or init_ckpt_dir(model_id, bits, group_size, rank)
    print(f"[init] INT{bits} g{group_size} rank{rank} alpha={args.alpha} calib={args.calib}\n"
          f"[init] base {gptq_path}\n[init] out  {out}")

    peft_model = build_qwha_model(gptq_path, rank=rank, scale=args.scale, device="cuda")

    # 1. quantization error, per layer, on the CPU
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="cpu").eval()
    with torch.inference_mode():
        up.compute_quant_error(peft_model, base, quant_method="gptq")

    # 2. the calibration pass, with the fp32 base alone on the card
    #
    # X^T X is accumulated on the UNQUANTIZED fp32 model, so the roots depend on the model and
    # the calibration set ONLY -- not on bits or group size. INT2 g32 and INT3 g64 want exactly
    # the same 224 matrices, and this pass costs ~3 h (each batch ships ~28 GB of [M,M] blocks
    # back to a float64 CPU accumulator, which is what keeps 57 GB of Hessian off a 40 GB card).
    # So it is computed once per (model, calibration set) and cached: the second width, and any
    # later rank or alpha sweep, reads it back instead of re-earning it.
    root_cache = os.path.join(os.path.dirname(gptq_path.rstrip("/")),
                              f"xtxroot_{os.path.basename(model_id)}_{args.calib}.pt")
    if os.path.isfile(root_cache):
        roots = torch.load(root_cache, map_location="cpu", weights_only=True)
        print(f"[init] loaded {len(roots)} cached xtx_root from {root_cache}")
        del base
        gc.collect(); torch.cuda.empty_cache()
    else:
        peft_model = peft_model.cpu()
        gc.collect(); torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, model_max_length=(cfg["model"]["max_seq_len"] if cfg else 2048),
            padding_side="right")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        base = base.cuda()
        if args.calib == "balanced":
            calibration_pass_bcal(base, cfg, tokenizer, args.calib_batch_size)
        else:
            calibration_pass_wikitext2(base, tokenizer)
        # The base's WEIGHTS are finished with here -- the eigendecompositions read only the
        # X^T X buffers hanging off its modules. Dropping them keeps this stage inside a 1-GPU
        # job's host memory: fp32 base (27 GB) + per-layer quantization errors (27 GB) +
        # float64 X^T X (57 GB) is already ~111 GB on a 7B, against the 110 GB it is given.
        for param in base.parameters():
            param.data = torch.empty(0)
        base = base.cpu()
        gc.collect(); torch.cuda.empty_cache()

        # 3. roots
        compute_roots(base, device="cuda")
        roots = {n: m.xtx_root for n, m in base.named_modules() if hasattr(m, "xtx_root")}
        del base
        gc.collect(); torch.cuda.empty_cache()
        # Written through a temporary file: a half-flushed 28 GB cache read back as roots would
        # be a silently wrong initialization, not a crash.
        tmp = root_cache + f".tmp{os.getpid()}"
        torch.save(roots, tmp)
        os.replace(tmp, root_cache)
        print(f"[init] cached {len(roots)} xtx_root to {root_cache}")

    # 3b. hand the roots to the adapted modules
    peft_model = peft_model.cuda()
    matched = 0
    for name, module in peft_model.named_modules():
        key = name.replace("base_model.model.", "", 1)
        if key in roots:
            module.register_buffer("xtx_root", roots[key])
            matched += 1
    print(f"[init] matched xtx_root on {matched} adapted layers")
    if matched != len(roots):
        raise SystemExit(f"only {matched} of {len(roots)} roots landed on an adapted layer")
    del roots
    gc.collect(); torch.cuda.empty_cache()

    # 4. AdaAlloc selection + refinement (upstream, unchanged)
    # The quantized base has been on the GPU since build_qwha_model(device="cuda"); the ONLY
    # things a blanket peft_model.cuda() would add are the 224 xtx_root (28 GB) and quant_error
    # (27 GB) tensors -- 55 GB against a 40 GB card. Upstream's loop moves each of them to the
    # GPU itself, one layer at a time, and frees it before the next (initialize.py:293,301,334),
    # so they belong on the CPU. Leaving them there is what makes this stage fit.
    with torch.inference_mode():
        up.initialize_adapter(peft_model, bits=bits, lora_rank=rank)

    os.makedirs(out, exist_ok=True)
    peft_model.save_pretrained(out)
    print(f"[init] saved {out}")
    print(f"[init] peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
