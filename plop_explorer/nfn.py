"""NFN (Normalized Feature Norm) scores, after Hayou, Ghosh & Yu, "PLoP" (arXiv:2506.20629).

For a linear module with weight W (d_out x d_in) and inputs z_t, with z_hat = z / ||z||:

    NFN(W) = mean_t ||W z_hat_t||  /  (||W||_F / sqrt(d_in))

The denominator is the RMS gain of W on a uniformly random unit vector, since
E||W u||^2 = ||W||_F^2 / d_in. So NFN = 1 means the task's inputs are no more
aligned with W than random directions; NFN > 1 means W already amplifies them,
and PLoP recommends placing LoRA adapters on the module types with the LOWEST
scores.

The reference implementation estimates the denominator by sampling one Gaussian
vector per token; the closed form above is its exact expectation (up to Jensen's
gap, which is O(1/d_out)) and removes that sampling noise.
"""

import math
import re
from dataclasses import dataclass, field

import torch

# Canonical module types (paper's names) -> leaf module names across architectures.
MODULE_TYPES = {
    "Query": ("q_proj",),
    "Key": ("k_proj",),
    "Value": ("v_proj",),
    "OutProj": ("o_proj",),
    "GateProj": ("gate_proj",),
    "UpProj": ("up_proj",),
    "DownProj": ("down_proj",),
}
_LEAF_TO_TYPE = {leaf: t for t, leaves in MODULE_TYPES.items() for leaf in leaves}
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def module_type(name):
    """Canonical type for a module name, or None if it is not a scored type."""
    return _LEAF_TO_TYPE.get(name.rsplit(".", 1)[-1])


def layer_index(name):
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


@dataclass
class _Accumulator:
    type: str
    layer: int
    random_gain: float  # ||W||_F / sqrt(d_in)
    d_in: int
    d_out: int
    norm_sum: float = 0.0
    tokens: int = 0


@dataclass
class NFNResult:
    modules: dict = field(default_factory=dict)  # name -> {type, layer, nfn, actual, random, tokens}

    def by_type(self):
        """Mean NFN per module type (unweighted over layers, as in the paper)."""
        out = {}
        for t in MODULE_TYPES:
            vals = [m["nfn"] for m in self.modules.values() if m["type"] == t]
            if vals:
                out[t] = sum(vals) / len(vals)
        return out

    def grid(self):
        """{type: [nfn per layer]} for the module-type x layer heatmap."""
        n_layers = 1 + max(m["layer"] for m in self.modules.values())
        out = {}
        for m in self.modules.values():
            out.setdefault(m["type"], [math.nan] * n_layers)[m["layer"]] = m["nfn"]
        return {t: out[t] for t in MODULE_TYPES if t in out}

    def recommend(self, k=3):
        """The k module types PLoP would adapt: lowest aggregate NFN."""
        scores = self.by_type()
        return sorted(scores, key=scores.get)[:k]


@torch.no_grad()
def score_model(model, batches, mask_padding=True):
    """Run `batches` (tokenizer outputs) through `model` and return NFN scores.

    mask_padding=False reproduces the reference code, which averages over pad
    positions as well as real tokens.
    """
    accs, hooks = {}, []
    current_mask = {}

    def make_hook(name):
        acc = accs[name]

        def hook(module, args, output):
            z = args[0].reshape(-1, acc.d_in).float()
            mask = current_mask.get("flat")
            if mask is not None:
                z = z[mask]
            z_hat = z / (z.norm(dim=1, keepdim=True) + 1e-8)
            y = z_hat @ module.weight.float().t()
            acc.norm_sum += y.norm(dim=1).sum().item()
            acc.tokens += z.shape[0]

        return hook

    for name, module in model.named_modules():
        t = module_type(name)
        if t is None or not isinstance(module, torch.nn.Linear):
            continue
        w = module.weight.float()
        accs[name] = _Accumulator(
            type=t,
            layer=layer_index(name),
            random_gain=w.norm().item() / math.sqrt(w.shape[1]),
            d_in=w.shape[1],
            d_out=w.shape[0],
        )
        hooks.append(module.register_forward_hook(make_hook(name)))
    if not accs:
        raise ValueError("No scorable modules found; unsupported architecture?")

    device = next(model.parameters()).device
    try:
        for batch in batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            current_mask["flat"] = batch["attention_mask"].reshape(-1).bool() if mask_padding else None
            model(**batch)
    finally:
        for h in hooks:
            h.remove()

    result = NFNResult()
    for name, a in accs.items():
        nfn = a.norm_sum / a.tokens / a.random_gain
        # "actual"/"random" use the reference JSON's scale (W scaled to unit RMS,
        # both divided by sqrt(d_in)), on which random = sqrt(d_out / d_in).
        random = math.sqrt(a.d_out / a.d_in)
        result.modules[name] = {
            "type": a.type,
            "layer": a.layer,
            "nfn": nfn,
            "actual": nfn * random,
            "random": random,
            "tokens": a.tokens,
        }
    return result
