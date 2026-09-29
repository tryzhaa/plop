"""Check PLoP's module picks against the sets the paper reports (gate G3).

    python scripts/check_selection.py                 # all models, all tasks
    python scripts/check_selection.py --models Qwen/Qwen3-0.6B --tasks math

The paper never says which prompt set produced each pick, so every task is
scored and the table shows which ones reproduce the reported set.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from plop_explorer.data import TASKS, load_prompts, make_batches  # noqa: E402
from plop_explorer.nfn import score_model  # noqa: E402

# Lowest-3 (PLoP) and highest-3 (PLoP^-1) as reported; None = not reported.
TARGETS = {
    "unsloth/Llama-3.2-1B-Instruct": ({"Value", "OutProj", "DownProj"}, None, "Fig 7, Sec 5"),
    "Qwen/Qwen2.5-0.5B": ({"Value", "OutProj", "UpProj"}, None, "Fig 7"),
    "Qwen/Qwen3-0.6B": ({"DownProj", "UpProj", "Value"}, {"GateProj", "Key", "Query"}, "Table 1"),
    "Qwen/Qwen3-1.7B": ({"DownProj", "OutProj", "Value"}, {"GateProj", "Key", "Query"}, "Tables 2-3"),
    "unsloth/gemma-3-1b-pt": ({"Key", "Value", "UpProj"}, {"OutProj", "GateProj", "DownProj"}, "App C.6"),
    "unsloth/gemma-3-1b-it": ({"Key", "Value", "UpProj"}, {"OutProj", "GateProj", "DownProj"}, "App C.6"),
}


def load(model_id, device, dtype):
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()


def score_pair(model, tok, prompts):
    # Reference setup: EOS right-padding, pad positions counted.
    tok.pad_token = tok.eos_token
    ref = score_model(model, make_batches(prompts, tok, padding_side="right"), mask_padding=False)
    masked = score_model(model, make_batches(prompts, tok), mask_padding=True)
    return ref, masked


def fmt(s):
    return ",".join(sorted(t.replace("Proj", "") for t in s))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(TARGETS))
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--nbsamples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    device = args.device
    dtype = torch.float16 if device == "mps" else torch.float32
    rows = []
    for model_id in args.models:
        tok = AutoTokenizer.from_pretrained(model_id)
        model, model_dtype = load(model_id, device, dtype), dtype
        for task in args.tasks:
            prompts = load_prompts(task, args.nbsamples, args.seed)
            t0 = time.time()
            try:
                ref, masked = score_pair(model, tok, prompts)
            except FloatingPointError:
                if model_dtype == torch.float32:
                    raise
                print(f"{model_id}: overflow in {model_dtype}, reloading in float32", flush=True)
                del model
                gc.collect()
                model, model_dtype = load(model_id, device, torch.float32), torch.float32
                ref, masked = score_pair(model, tok, prompts)
            scores = ref.by_type()
            name = model_id.split("/")[-1]
            out = ROOT / "results" / f"{name}_{task}.json"
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps({"padded": ref.modules, "masked": masked.modules}, indent=1))
            rows.append((name, task, scores, set(ref.recommend(3)), set(masked.recommend(3)), model_id))
            print(f"{name}/{task}: {time.time() - t0:.0f}s", flush=True)
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    print(f"\n{'model':<24}{'task':<9}{'PLoP ref':<18}{'PLoP mask':<18}{'top-3 (PLoP^-1)':<18}paper PLoP / PLoP^-1")
    for name, task, scores, low_ref, low_mask, model_id in rows:
        want_low, want_high, src = TARGETS.get(model_id, (None, None, ""))
        high = set(sorted(scores, key=scores.get)[-3:])

        def mark(got, want):
            return fmt(got) + (" ✓" if got == want else " ✗" if want else "")

        print(
            f"{name:<24}{task:<9}{mark(low_ref, want_low):<18}{mark(low_mask, want_low):<18}"
            f"{mark(high, want_high):<18}{fmt(want_low) if want_low else '-'} / {fmt(want_high) if want_high else '-'} ({src})"
        )
    print("\nPer-type scores (reference setup):")
    for name, task, scores, *_ in rows:
        print(f"{name:<24}{task:<9}" + " ".join(f"{t[:4]}={v:.2f}" for t, v in sorted(scores.items(), key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()
