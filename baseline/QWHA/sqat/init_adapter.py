"""
QWHA adapter initialization (AdaAlloc parameter selection + value refinement) for one cell.

Upstream's `src/init/initialize.py` in three stages -- quantization error, a wikitext2 calibration
pass that accumulates X^T X per projection, then the per-layer spectrum selection -- all called
from upstream unchanged. Two things differ, both about the 40 GB card:

  1. Upstream's `calibration_forward` runs the eigendecompositions at the end of the pass, while
     the fp32 base (27 GB) is still resident. Here the pass and the roots are separate steps, with
     the base moved back to the host in between; the quantized model likewise leaves the GPU for
     the duration of the pass. The arithmetic (same damping, same clamp, same R) is copied
     verbatim from upstream.
  2. Paths carry the group size, so INT3 g64 and INT2 g32 do not overwrite each other.
"""

import argparse
import gc
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# qwha_common first: importing it is what puts upstream's src/ and src/init/ on sys.path.
from qwha_common import build_qwha_model, gptq_base_dir, init_ckpt_dir
import initialize as up          # upstream src/init/initialize.py


def calibration_pass(model, tokenizer, dataset_id: str = "wikitext2", batch_size: int = 1):
    """Upstream `calibration_forward`, minus its trailing call to the root computation."""
    max_length = min(model.config.max_position_embeddings, 4096)
    up.register_hook(model)

    if dataset_id != "wikitext2":
        raise NotImplementedError(f"calibration dataset {dataset_id}")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_id", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("-b", "--bits", type=int, required=True)
    ap.add_argument("-g", "--group_size", type=int, required=True)
    ap.add_argument("-r", "--rank", type=int, default=64)
    ap.add_argument("-s", "--scale", type=float, default=0.25,
                    help="scaling the spectrum is STORED at; training replays it at its own "
                         "scale (upstream's initialize.py default, kept)")
    ap.add_argument("-a", "--alpha", type=float, default=1.0,
                    help="AdaAlloc exponent; upstream default 1.0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    up.ALPHA = args.alpha
    out = args.out or init_ckpt_dir(args.model_id, args.bits, args.group_size, args.rank)
    gptq_path = gptq_base_dir(args.model_id, args.bits, args.group_size)
    print(f"[init] INT{args.bits} g{args.group_size} rank{args.rank} alpha={args.alpha}\n"
          f"[init] base {gptq_path}\n[init] out  {out}")

    peft_model = build_qwha_model(gptq_path, rank=args.rank, scale=args.scale, device="cuda")

    # The order below is memory-driven and nothing else. A 1-GPU PBS slot on this cluster is a
    # quarter node: 110 GB of host RAM and one 40 GB card. Upstream's order holds the fp32 base
    # (27 GB), the per-layer quantization errors (27 GB) and the float64 X^T X buffers (57 GB)
    # at the same time, which does not fit. Running the calibration FIRST and the quantization
    # error afterwards means the errors only ever coexist with the roots (~55 GB), and the base
    # never has to come back to the host at all.

    # 1. wikitext2 pass with the fp32 base alone on the card
    peft_model = peft_model.cpu()
    gc.collect(); torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float32, device_map="cpu").eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    with torch.inference_mode():
        calibration_pass(base, tokenizer, "wikitext2")

    # 2. roots, then hand them to the adapted modules (each X^T X freed as its root is taken)
    compute_roots(base, device="cuda")
    roots = {n: m.xtx_root for n, m in base.named_modules() if hasattr(m, "xtx_root")}
    matched = 0
    for name, module in peft_model.named_modules():
        key = name.replace("base_model.model.", "", 1)
        if key in roots:
            module.register_buffer("xtx_root", roots[key])
            matched += 1
    print(f"[init] matched xtx_root on {matched} adapted layers")
    del roots

    # 3. quantization error, per layer, against the base still resident on the card
    peft_model = peft_model.cuda()
    with torch.inference_mode():
        up.compute_quant_error(peft_model, base, quant_method="gptq")
    del base
    gc.collect(); torch.cuda.empty_cache()

    # 4. AdaAlloc selection + refinement (upstream, unchanged)
    peft_model = peft_model.cuda()
    for _, module in peft_model.named_modules():
        if hasattr(module, "qweight"):     # plain attributes: .cuda() does not move them
            module.wf_unsqueeze_zero = module.wf_unsqueeze_zero.cuda()
            module.wf_unsqueeze_neg_one = module.wf_unsqueeze_neg_one.cuda()
    with torch.inference_mode():
        up.initialize_adapter(peft_model, bits=args.bits, lora_rank=args.rank)

    os.makedirs(out, exist_ok=True)
    peft_model.save_pretrained(out)
    print(f"[init] saved {out}")
    print(f"[init] peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
