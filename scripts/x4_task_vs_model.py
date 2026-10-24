"""X4: does NFN depend on the task, or mostly on the model?

    python scripts/x4_task_vs_model.py                    # 5 models x (4 tasks + 3 controls)
    python scripts/x4_task_vs_model.py --models Qwen/Qwen3-0.6B

Scores each model on the four task prompt sets and three controls (random tokens,
shuffled words, plain prose), padding masked, and writes one stamped file per
model to reference/ours/x4/. Analysis: scripts/x4_analyze.py.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from plop_explorer.data import CONTROLS, TASKS, load_control, load_prompts, make_batches  # noqa: E402
from plop_explorer.nfn import score_model  # noqa: E402
from plop_explorer.runlog import save_result  # noqa: E402

MODELS = [
    "unsloth/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "unsloth/gemma-3-1b-it",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--nbsamples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    for model_id in args.models:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
        scores = {}
        for name in [*TASKS, *CONTROLS]:
            t0 = time.time()
            prompts = (load_prompts(name, args.nbsamples, args.seed) if name in TASKS
                       else load_control(name, args.nbsamples, args.seed, tok))
            scores[name] = score_model(model, make_batches(prompts, tok), mask_padding=True).by_type()
            print(f"{model_id}/{name}: {time.time() - t0:.0f}s", flush=True)
        stem = model_id.split("/")[-1]
        save_result(ROOT / "reference/ours/x4" / f"{stem}.json", scores, model=model_id,
                    nbsamples=args.nbsamples, seed=args.seed, device="cpu", dtype="float32", mask_padding=True)
        del model
        gc.collect()


if __name__ == "__main__":
    main()
