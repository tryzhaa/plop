"""Compare our NFN scores with the PLoP authors' published results.

    python scripts/check_reference.py --task math

Runs with padding included (as the reference does) and masked (our default),
and prints both against the reference per-type scores.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plop_explorer.data import load_prompts, make_batches  # noqa: E402
from plop_explorer.nfn import MODULE_TYPES, score_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "reference/plop_reference_by_type.json").read_text())


def main():
    p = argparse.ArgumentParser()
    # Ungated mirror of meta-llama/Llama-3.2-1B-Instruct (same weights).
    p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--task", default="math", choices=["math", "code", "history", "logic"])
    p.add_argument("--nbsamples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device, dtype = ("mps", torch.float16) if torch.backends.mps.is_available() else ("cpu", torch.float32)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    batches = make_batches(load_prompts(args.task, args.nbsamples, args.seed), tok)

    runs = {}
    for mask in (False, True):
        t0 = time.time()
        runs[mask] = score_model(model, batches, mask_padding=mask)
        print(f"mask_padding={mask}: {time.time() - t0:.1f}s")

    ref = REFERENCE["Llama-3.2-1B-Instruct"][args.task]
    print(f"\n{args.model} / {args.task}  (device={device}, n={args.nbsamples}, seed={args.seed})")
    print(f"{'type':<10}{'reference':>10}{'ours:pad':>10}{'diff':>8}{'ours:mask':>11}{'diff':>8}")
    for t in MODULE_TYPES:
        r, a, b = ref[t], runs[False].by_type()[t], runs[True].by_type()[t]
        print(f"{t:<10}{r:>10.3f}{a:>10.3f}{a - r:>+8.3f}{b:>11.3f}{b - r:>+8.3f}")
    ref_top3 = sorted(ref, key=ref.get)[:3]
    print(f"\nPLoP top-3  reference: {ref_top3}")
    print(f"            ours:pad:  {runs[False].recommend()}")
    print(f"            ours:mask: {runs[True].recommend()}")

    out = ROOT / "results" / f"{args.model.split('/')[-1]}_{args.task}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"padded": runs[False].modules, "masked": runs[True].modules}, indent=1))
    print(f"\nSaved per-module scores to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
