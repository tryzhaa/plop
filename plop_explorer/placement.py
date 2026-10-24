"""Turn a placement name into module types, and wrap a model with LoRA on exactly those types."""

import json
from pathlib import Path

from peft import LoraConfig, TaskType, get_peft_model

from plop_explorer.lora_params import PLACEMENTS, count
from plop_explorer.nfn import MODULE_TYPES

OURS = Path(__file__).resolve().parents[1] / "reference" / "ours"


def resolve_types(placement, model_name=None, task="math"):
    """`attn`, `mlp`, `all`; `plop` / `plop_inv` from our committed scores; `custom:Query,Key,...`."""
    key = placement.lower()
    named = {k.lower(): v for k, v in PLACEMENTS.items()}
    if key in named:
        return named[key]
    if key in ("plop", "plop_inv"):
        stem = model_name.split("/")[-1]
        scores = json.loads((OURS / f"{stem}_{task}.json").read_text())["result"]["reference"]
        return tuple(scores[key])
    if key.startswith("custom:"):
        types = tuple(placement.split(":", 1)[1].split(","))
        unknown = set(types) - set(MODULE_TYPES)
        if unknown:
            raise ValueError(f"Unknown module types {sorted(unknown)}; expected {list(MODULE_TYPES)}")
        return types
    raise ValueError(f"Unknown placement {placement!r}")


def target_modules(types):
    return [leaf for t in types for leaf in MODULE_TYPES[t]]


def build_lora_model(model, types, rank, dropout=0.1):
    """LoRA (alpha = 2r) on `types` of a sequence-classification model; the new `score` head is trained too.

    Asserts the adapter parameter count equals `lora_params.count`, the counter used to match budgets.
    """
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=2 * rank, lora_dropout=dropout,
                     target_modules=target_modules(types), modules_to_save=["score"])
    model = get_peft_model(model, cfg)
    adapter = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n and p.requires_grad)
    expected = count(model.config, types, rank)
    assert adapter == expected, f"adapter parameters {adapter} != lora_params.count {expected}"
    return model, adapter
