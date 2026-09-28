import math

import torch

from plop_explorer.nfn import layer_index, module_type, score_model


class Block(torch.nn.Module):
    def __init__(self, d, w_q=None):
        super().__init__()
        self.q_proj = torch.nn.Linear(d, d, bias=False)
        self.v_proj = torch.nn.Linear(d, d // 2, bias=False)
        if w_q is not None:
            self.q_proj.weight.data = w_q

    def forward(self, x):
        return self.q_proj(x) + self.v_proj(x).repeat(1, 1, 2)


class Tiny(torch.nn.Module):
    """Stands in for a HF model: named `model.layers.N.*`, takes input_ids + attention_mask."""

    def __init__(self, d=64, n_layers=2, w_q=None, inputs=None):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(Block(d, w_q) for _ in range(n_layers))
        self.inputs = inputs
        self.d = d

    def forward(self, input_ids, attention_mask, use_cache=None):
        x = self.inputs if self.inputs is not None else torch.randn(*input_ids.shape, self.d)
        for layer in self.model.layers:
            x = layer(x)
        return x


def batch(b=2, t=5, pad=0):
    mask = torch.ones(b, t, dtype=torch.long)
    if pad:
        mask[:, -pad:] = 0
    return {"input_ids": torch.zeros(b, t, dtype=torch.long), "attention_mask": mask}


def test_names():
    assert module_type("model.layers.3.self_attn.q_proj") == "Query"
    assert module_type("model.layers.3.mlp.down_proj") == "DownProj"
    assert module_type("lm_head") is None
    assert layer_index("model.layers.12.mlp.up_proj") == 12


def test_random_inputs_score_near_one():
    torch.manual_seed(0)
    res = score_model(Tiny(d=256), [batch(b=8, t=64)])
    for m in res.modules.values():
        assert abs(m["nfn"] - 1) < 0.05, m


def test_aligned_inputs_score_above_one():
    # Inputs along W's top right-singular vector get the top singular value as gain.
    torch.manual_seed(0)
    d = 64
    w = torch.randn(d, d)
    top = torch.linalg.svd(w).Vh[0]
    x = top.expand(2, 5, d).clone()
    res = score_model(Tiny(d=d, n_layers=1, w_q=w, inputs=x), [batch()])
    s = torch.linalg.svdvals(w)
    expected = (s[0] / (w.norm() / math.sqrt(d))).item()
    assert math.isclose(res.modules["model.layers.0.q_proj"]["nfn"], expected, rel_tol=1e-4)
    assert res.recommend(1) == ["Value"]


def test_padding_mask_excludes_pad_tokens():
    torch.manual_seed(0)
    masked = score_model(Tiny(), [batch(pad=2)], mask_padding=True)
    unmasked = score_model(Tiny(), [batch(pad=2)], mask_padding=False)
    assert all(m["tokens"] == 6 for m in masked.modules.values())
    assert all(m["tokens"] == 10 for m in unmasked.modules.values())


def test_grid_shape():
    res = score_model(Tiny(n_layers=3), [batch()])
    grid = res.grid()
    assert list(grid) == ["Query", "Value"]
    assert all(len(v) == 3 for v in grid.values())


def test_exact_baseline_matches_sampled_baseline():
    # The reference samples a random unit vector u per token and averages ||W u||;
    # our closed form ||W||_F / sqrt(d_in) is sqrt(E||W u||^2), equal up to Jensen's gap.
    torch.manual_seed(0)
    w = torch.randn(96, 64) * torch.linspace(0.1, 3, 64)  # anisotropic columns
    u = torch.randn(200_000, 64)
    u = u / u.norm(dim=1, keepdim=True)
    sampled = (u @ w.t()).norm(dim=1).mean().item()
    exact = (w.norm() / math.sqrt(64)).item()
    assert math.isclose(sampled, exact, rel_tol=0.01)
