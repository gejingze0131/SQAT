#!/usr/bin/env python
"""Teacher-forced probe of the ANSWER TOKEN on held-out commonsense records.

WHY. Under data.loss_span=instruction+response the training loss is ~95% question text, so
whether the answer token is being learned is invisible in the logged curve; and under
response-only the logged loss is ~7/8 template. Both hide the one number the INT2 plateau story
is about. This probe reads it directly: for each held-out record it teacher-forces the prompt +
response and reports
    resp_loss   mean CE over the response tokens (the "response-only" loss, comparable across
                loss_span cells)
    ans_loss    CE at the answer token alone (the token before the trailing newline:
                "answer2", "true", "ending4", ...)
    ans_acc     greedy argmax == gold at that position -- accuracy WITHOUT generation
    majority    share of records whose gold answer is the task's most common answer, so ans_acc
                can be read against the degenerate solution
per task and overall. One model build per base; each checkpoint is overlaid with
load_saltq_trainable, so probing N checkpoints costs one build + N overlays.

Usage
-----
  python scripts/probe_answer_token.py --config configs/X.yaml \
      --saltq_base_dir outputs/.../saltq_base_2bit_g32 \
      --checkpoints outputs/A/final outputs/B/checkpoint-500 ... \
      [--per_task 64] [--batch 8] [--out results/probe/X.json]
A checkpoint path of "base" probes the frozen base itself (step 0).
"""
import argparse, collections, json, os, sys
import torch, torch.nn.functional as F, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import PROMPT, IGNORE_INDEX
from src.qat_saltq import build_saltq_model, load_saltq_trainable
from src.permute_common import register_boundary_gathers_from_meta
from transformers import AutoTokenizer


def records(path, per_task, seed=0):
    d = json.load(open(path))
    by = collections.defaultdict(list)
    for r in d: by[r['type']].append(r)
    g = torch.Generator().manual_seed(seed)
    out = []
    for t in sorted(by):
        idx = torch.randperm(len(by[t]), generator=g)[:per_task].tolist()
        out += [by[t][i] for i in idx]
    return out, {t: collections.Counter(r['output'] for r in rs).most_common(1)[0][0] for t, rs in by.items()}


def encode(rec, tok):
    src = rec['instruction'] if rec['instruction'].startswith(PROMPT[:40]) \
        else PROMPT.format_map(dict(instruction=rec['instruction']))
    # test.json ships bare answers ("answer3"); training targets are "the correct answer is
    # answer3" (src/data.py record adapters). Rebuild the training-format response so the
    # teacher-forced conditional matches what the model was trained on.
    out = rec['output'] if rec['output'].startswith('the correct answer is') \
        else f"the correct answer is {rec['output']}"
    tgt = f"{out}\n{tok.eos_token}"
    full = tok(src + tgt, truncation=True, max_length=2048).input_ids
    n_src = len(tok(src, truncation=True, max_length=2048).input_ids)
    labels = [IGNORE_INDEX] * n_src + full[n_src:]
    # answer token = last response token before the newline token(s) and EOS
    resp = full[n_src:]
    nl = tok("\n", add_special_tokens=False).input_ids
    k = len(resp) - 1
    while k > 0 and (resp[k] == tok.eos_token_id or resp[k] in nl or tok.decode([resp[k]]).strip() == ""):
        k -= 1
    return full, labels, n_src + k


@torch.no_grad()
def probe(model, tok, recs, batch, device):
    per = collections.defaultdict(lambda: dict(resp_loss=0., ans_loss=0., ans_acc=0., n=0, ntok=0))
    for i in range(0, len(recs), batch):
        chunk = recs[i:i + batch]
        enc = [encode(r, tok) for r in chunk]
        L = max(len(e[0]) for e in enc)
        ids = torch.full((len(enc), L), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(enc), L), IGNORE_INDEX, dtype=torch.long)
        for j, (f, l, _) in enumerate(enc):
            ids[j, :len(f)] = torch.tensor(f); lab[j, :len(l)] = torch.tensor(l)
        ids, lab = ids.to(device), lab.to(device)
        logits = model(input_ids=ids, attention_mask=ids.ne(tok.pad_token_id)).logits.float()
        lp = F.log_softmax(logits[:, :-1], -1)
        tgt = lab[:, 1:]
        for j, (r, (_, _, apos)) in enumerate(zip(chunk, enc)):
            m = tgt[j] != IGNORE_INDEX
            tok_lp = lp[j].gather(-1, tgt[j].clamp(min=0).unsqueeze(-1)).squeeze(-1)
            resp_loss = -(tok_lp[m]).mean().item()
            a = apos - 1                                   # logits at a predict token apos
            ans_loss = -tok_lp[a].item()
            ans_acc = float(lp[j, a].argmax().item() == tgt[j, a].item())
            s = per[r['type']]
            s['resp_loss'] += resp_loss; s['ans_loss'] += ans_loss; s['ans_acc'] += ans_acc; s['n'] += 1
    return {t: {k: (v / s['n'] if k != 'n' else v) for k, v in s.items() if k != 'ntok'} for t, s in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--saltq_base_dir', required=True)
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--test_json', default='datasets/commonsense/test.json')
    ap.add_argument('--per_task', type=int, default=64); ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); sq = cfg['qat'].get('saltq', {}) or {}
    device = 'cuda'
    tok = AutoTokenizer.from_pretrained(a.saltq_base_dir)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model, meta = build_saltq_model(
        a.saltq_base_dir, dtype=torch.bfloat16, gradient_checkpointing=False,
        train_scale=bool(sq.get('train_scale', False)), continuous_z=bool(sq.get('continuous_z', True)),
        zplora_rank=int(sq.get('zplora_rank', 0)), zplora_alpha=float(sq.get('zplora_alpha', 16.0)),
        salient_lora=bool(sq.get('salient_lora', False)), slora_alpha=sq.get('slora_alpha', None))
    # REQUIRED for any forward of the multi-segment permuted model: the residual stream is
    # re-ordered at each segment boundary by runtime hooks that training installs in
    # prepare_model and eval installs on the exported model. Without them the first probe
    # returned resp_loss 11.3 on a base that scores 50.35 generatively -- i.e. random logits.
    hooks = register_boundary_gathers_from_meta(model, meta['perm_meta'])
    print(f"[probe] registered {len(hooks)} boundary gathers")
    model = model.to(device).eval()
    recs, majority = records(a.test_json, a.per_task)
    maj_share = {t: sum(r['output'] == majority[t] for r in recs if r['type'] == t) / max(1, sum(r['type'] == t for r in recs)) for t in majority}
    results = {}
    for ck in a.checkpoints:
        if ck != 'base':
            load_saltq_trainable(model, ck)
        res = probe(model, tok, recs, a.batch, device)
        tasks = sorted(res)
        overall = {k: sum(res[t][k] for t in tasks) / len(tasks) for k in ('resp_loss', 'ans_loss', 'ans_acc')}
        results[ck] = dict(per_task=res, overall=overall)
        print(f"\n== {ck}")
        print(f"{'task':14s} {'resp_loss':>9s} {'ans_loss':>9s} {'ans_acc':>8s} {'majority':>9s}")
        for t in tasks:
            print(f"{t:14s} {res[t]['resp_loss']:9.4f} {res[t]['ans_loss']:9.4f} {res[t]['ans_acc']:8.3f} {maj_share[t]:9.3f}")
        print(f"{'MEAN':14s} {overall['resp_loss']:9.4f} {overall['ans_loss']:9.4f} {overall['ans_acc']:8.3f}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(dict(results=results, majority_share=maj_share, per_task=a.per_task), open(a.out, 'w'), indent=1)
        print(f"wrote {a.out}")


if __name__ == '__main__':
    main()
