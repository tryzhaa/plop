"""X5 (how many prompts are enough?) and X6 (base vs instruction-tuned).

    python scripts/x5_x6.py x5     # n in {5,10,25,50,100} x seeds 0-4, math, 3 models
    python scripts/x5_x6.py x6     # base/instruct pairs on math

Both score with padding masked on CPU in float32 and write stamped files to reference/ours/.
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from plop_explorer.data import load_prompts, make_batches  # noqa: E402
from plop_explorer.nfn import score_model  # noqa: E402
from plop_explorer.runlog import save_result  # noqa: E402

X5_MODELS = ["unsloth/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B"]
X5_NS, X5_SEEDS = [5, 10, 25, 50, 100], range(5)
# A fixed pool: the reference's 2n-question pool would make small-n subsets overlap and hide the spread.
X5_POOL = 1000
# (base, instruct) pairs; Gemma's pair is already in reference/ours from G3 but is rescored here for one setup.
X6_PAIRS = [
    ("unsloth/gemma-3-1b-pt", "unsloth/gemma-3-1b-it"),
    ("Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B"),
    ("unsloth/Llama-3.2-1B", "unsloth/Llama-3.2-1B-Instruct"),
]


def score(model, tok, task, n, seed, pool_size=None):
    prompts = load_prompts(task, n, seed, pool_size)
    return score_model(model, make_batches(prompts, tok), mask_padding=True).by_type()


def load(model_id):
    return AutoTokenizer.from_pretrained(model_id), AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exp", choices=["x5", "x6"])
    args = p.parse_args()

    if args.exp == "x5":
        for model_id in X5_MODELS:
            tok, model = load(model_id)
            runs = {}
            for n in X5_NS:
                for seed in X5_SEEDS:
                    runs[f"n{n}_seed{seed}"] = score(model, tok, "math", n, seed, X5_POOL)
                    print(f"{model_id} n={n} seed={seed}", flush=True)
            save_result(ROOT / "reference/ours/x5" / f"{model_id.split('/')[-1]}.json", runs, model=model_id,
                        task="math", ns=X5_NS, seeds=list(X5_SEEDS), pool_size=X5_POOL, device="cpu", dtype="float32")
            del model
            gc.collect()
    else:
        for model_id in [m for pair in X6_PAIRS for m in pair]:
            tok, model = load(model_id)
            save_result(ROOT / "reference/ours/x6" / f"{model_id.split('/')[-1]}.json",
                        score(model, tok, "math", 100, 0), model=model_id, task="math", nbsamples=100, seed=0,
                        device="cpu", dtype="float32")
            print(model_id, flush=True)
            del model
            gc.collect()


if __name__ == "__main__":
    main()
