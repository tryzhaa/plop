"""LoRA parameter counts, for comparing placements at equal trainable-parameter budgets.

A rank-r adapter on a Linear(d_in -> d_out) has r * (d_in + d_out) parameters
(A is r x d_in, B is d_out x r). The paper's "Attn" placement is Query, Key and
Value only; that is the only reading under which its #Params columns add up.
"""

from plop_explorer.nfn import MODULE_TYPES

PLACEMENTS = {
    "Attn": ("Query", "Key", "Value"),
    "MLP": ("GateProj", "UpProj", "DownProj"),
    "All": tuple(MODULE_TYPES),
}


def module_shapes(config):
    """{type: (d_in, d_out)} for one decoder layer, from a HF config (or anything with the same attributes)."""
    h = config.hidden_size
    head_dim = getattr(config, "head_dim", None) or h // config.num_attention_heads
    q = config.num_attention_heads * head_dim
    kv = config.num_key_value_heads * head_dim
    m = config.intermediate_size
    return {
        "Query": (h, q),
        "Key": (h, kv),
        "Value": (h, kv),
        "OutProj": (q, h),
        "GateProj": (h, m),
        "UpProj": (h, m),
        "DownProj": (m, h),
    }


def per_rank(config, types):
    """Trainable parameters per unit of rank, summed over all layers."""
    shapes = module_shapes(config)
    return config.num_hidden_layers * sum(sum(shapes[t]) for t in types)


def count(config, types, rank):
    return rank * per_rank(config, types)


def match_rank(config, types, budget):
    """The rank whose parameter count is closest to `budget` (ties go to the lower rank)."""
    unit = per_rank(config, types)
    lo = max(1, int(budget // unit))
    return min((lo, lo + 1), key=lambda r: (abs(r * unit - budget), r))
