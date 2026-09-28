"""The paper's own #Params columns (Tables 1-3, App C.2) as targets for the counter."""

from types import SimpleNamespace

import pytest

from plop_explorer.lora_params import PLACEMENTS, count, match_rank


def cfg(h, layers, heads, kv, head_dim, mlp):
    return SimpleNamespace(
        hidden_size=h,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=head_dim,
        intermediate_size=mlp,
    )


# hidden, layers, heads, kv heads, head dim, MLP width (checked against each model's config.json)
QWEN25_05B = cfg(896, 24, 14, 2, 64, 4864)
LLAMA32_1B = cfg(2048, 16, 32, 8, 64, 8192)
QWEN3_06B = cfg(1024, 28, 16, 8, 128, 3072)
QWEN3_17B = cfg(2048, 28, 16, 8, 128, 6144)

ATTN, MLP, ALL = PLACEMENTS["Attn"], PLACEMENTS["MLP"], PLACEMENTS["All"]
D, G, U, K, Q, V, O = "DownProj", "GateProj", "UpProj", "Key", "Query", "Value", "OutProj"


@pytest.mark.parametrize(
    "config, types, rank, paper_millions",
    [
        # Table 1, Qwen3-0.6B
        (QWEN3_06B, ATTN, 64, 12.8),
        (QWEN3_06B, MLP, 64, 22.0),
        (QWEN3_06B, (D, U, V), 64, 18.4),
        (QWEN3_06B, (D, U, V), 76, 21.8),
        (QWEN3_06B, (G, K, Q), 64, 16.5),
        (QWEN3_06B, ALL, 64, 40.4),
        # Table 2, Qwen3-1.7B
        (QWEN3_17B, ATTN, 64, 18.4),
        (QWEN3_17B, MLP, 64, 44.0),
        (QWEN3_17B, (D, O, V), 64, 27.5),
        (QWEN3_17B, (D, O, V), 102, 43.9),
        (QWEN3_17B, ALL, 64, 69.7),
        # Table 3 (GRPO), Qwen3-1.7B
        (QWEN3_17B, ATTN, 16, 4.58),
        (QWEN3_17B, ATTN, 25, 7.17),
        (QWEN3_17B, MLP, 16, 11.01),
        (QWEN3_17B, (D, O, V), 16, 6.88),
        (QWEN3_17B, (D, O, V), 25, 10.75),
    ],
)
def test_matches_paper_param_counts(config, types, rank, paper_millions):
    decimals = len(str(paper_millions).split(".")[1])
    # Paper rounds; allow one unit in the last reported digit.
    assert abs(count(config, types, rank) / 1e6 - paper_millions) <= 10**-decimals


def test_anli_ranks_are_budget_matched():
    # App C.2: MLP r=8 is the budget; Attn and PLoP ranks roughly match it.
    budget = count(LLAMA32_1B, MLP, 8)
    assert match_rank(LLAMA32_1B, ATTN, budget) == 27
    assert match_rank(LLAMA32_1B, (V, O, D), budget) == 15
    budget = count(QWEN25_05B, MLP, 8)
    assert match_rank(QWEN25_05B, ATTN, budget) == 36
    # The paper used r=17 here, though r=16 is closer to the budget.
    assert match_rank(QWEN25_05B, (V, O, U), budget) == 16


def test_match_rank_hits_exact_multiple():
    assert match_rank(QWEN3_06B, MLP, count(QWEN3_06B, MLP, 64)) == 64
