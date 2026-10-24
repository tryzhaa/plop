"""E6: LoRA fine-tuning on ANLI under a chosen placement (plan Section 14.5).

    python experiments/e6_anli_train.py --model Qwen/Qwen2.5-0.5B --placement plop --seed 0 --steps 300   # pilot
    python experiments/e6_anli_train.py --model Qwen/Qwen2.5-0.5B --placement mlp --rank 8 --steps 5000

Placements: plop, plop_inv (from our committed scores), attn, mlp, all, custom:Query,Key,...
Unless --rank is given, the rank is matched to the MLP r=8 parameter budget with lora_params.match_rank.

Setup follows the paper (AdamW, linear schedule, no warmup, dropout 0.1, max length 256, alpha = 2r) except
precision: base weights stay fp32 and the forward pass uses fp16 autocast on CUDA, because T4/P100 lack bf16.
The paper does not give batch size, learning rate or evaluation frequency; the defaults below are ours.
Writes <out_dir>/<run>.csv (curve) and <out_dir>/<run>.json (stamped final result).
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from plop_explorer.lora_params import PLACEMENTS, count, match_rank  # noqa: E402
from plop_explorer.placement import build_lora_model, resolve_types  # noqa: E402
from plop_explorer.runlog import save_result  # noqa: E402

PAPER_RANKS = {  # Appendix C.2: (MLP, Attn, PLoP)
    "Qwen/Qwen2.5-0.5B": (8, 36, 17),
    "unsloth/Llama-3.2-1B": (8, 27, 15),
}


def anli(split):
    """All three ANLI rounds combined; `split` is train, dev or test."""
    return concatenate_datasets([load_dataset("facebook/anli", split=f"{split}_r{r}") for r in (1, 2, 3)])


def encode(tok, rows, max_len, device):
    text = [f"Premise: {p}\nHypothesis: {h}" for p, h in zip(rows["premise"], rows["hypothesis"])]
    batch = tok(text, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}, torch.tensor(rows["label"], device=device)


@torch.no_grad()
def evaluate(model, tok, ds, args, device):
    model.eval()
    correct = 0
    for i in range(0, len(ds), args.eval_batch):
        batch, labels = encode(tok, ds[i : i + args.eval_batch], args.max_len, device)
        with torch.autocast(device, dtype=torch.float16, enabled=args.amp):
            correct += (model(**batch).logits.argmax(-1) == labels).sum().item()
    model.train()
    return correct / len(ds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--placement", default="plop")
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_batch", type=int, default=64)
    p.add_argument("--dev_subset", type=int, default=1000)
    p.add_argument("--limit_eval", type=int, default=None, help="cap final dev/test sizes (smoke tests only)")
    p.add_argument("--out_dir", default=str(ROOT / "results" / "e6"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.amp = device == "cuda"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3, dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id

    types = resolve_types(args.placement, args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    budget = count(cfg, PLACEMENTS["MLP"], 8)
    rank = args.rank or match_rank(cfg, types, budget)
    model, adapter_params = build_lora_model(model, types, rank)
    model.to(device).train()
    print(f"placement={args.placement} types={types} rank={rank} adapter_params={adapter_params:,} "
          f"(MLP r=8 budget {budget:,}; paper ranks for this model, if known: {PAPER_RANKS.get(args.model)})")

    train, dev, test = anli("train"), anli("dev"), anli("test")
    if args.limit_eval:
        dev, test = dev.select(range(args.limit_eval)), test.select(range(args.limit_eval))
    dev_sub = dev.shuffle(seed=0).select(range(min(args.dev_subset, len(dev))))
    print(f"train {len(train)}, dev {len(dev)} (curve on {len(dev_sub)}), test {len(test)}")

    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=args.steps)
    scaler = torch.amp.GradScaler(enabled=args.amp)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed)).tolist()

    run = f"{args.model.split('/')[-1]}_{args.placement.replace(':', '-').replace(',', '+')}_r{rank}_s{args.seed}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    curve = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = [order[(step * args.batch + j) % len(order)] for j in range(args.batch)]
        batch, labels = encode(tok, train[idx], args.max_len, device)
        with torch.autocast(device, dtype=torch.float16, enabled=args.amp):
            loss = torch.nn.functional.cross_entropy(model(**batch).logits.float(), labels)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            acc = evaluate(model, tok, dev_sub, args, device)
            curve.append((step, loss.item(), acc))
            print(f"step {step}: loss {loss.item():.3f} dev_acc {acc:.3f} ({(time.time() - t0) / step:.2f}s/step)", flush=True)

    with open(out / f"{run}.csv", "w", newline="") as f:
        csv.writer(f).writerows([("step", "train_loss", "dev_acc"), *curve])
    result = {"test_acc": evaluate(model, tok, test, args, device), "dev_acc": evaluate(model, tok, dev, args, device),
              "curve_dev_acc": curve[-1][2], "adapter_params": adapter_params, "rank": rank, "types": list(types)}
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
    save_result(out / f"{run}.json", result, model=args.model, placement=args.placement, seed=args.seed, steps=args.steps,
                batch=args.batch, lr=args.lr, device=device, amp_fp16=args.amp, seconds_per_step=(time.time() - t0) / args.steps,
                peak_gpu_gib=peak)
    print(f"final: test_acc {result['test_acc']:.4f} dev_acc {result['dev_acc']:.4f}")


if __name__ == "__main__":
    main()
