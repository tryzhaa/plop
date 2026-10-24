import pytest
import torch
from transformers import AutoModelForSequenceClassification, Qwen2Config

from plop_explorer.lora_params import PLACEMENTS
from plop_explorer.placement import build_lora_model, resolve_types, target_modules


def tiny():
    cfg = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=100, num_labels=3, pad_token_id=0)
    return AutoModelForSequenceClassification.from_config(cfg)


def test_resolve_named_and_custom():
    assert resolve_types("attn") == PLACEMENTS["Attn"]
    assert resolve_types("custom:Value,OutProj") == ("Value", "OutProj")
    with pytest.raises(ValueError):
        resolve_types("custom:Bogus")
    with pytest.raises(ValueError):
        resolve_types("nonsense")


def test_plop_from_committed_scores():
    assert set(resolve_types("plop", "Qwen/Qwen2.5-0.5B")) == {"Value", "OutProj", "UpProj"}
    assert set(resolve_types("plop_inv", "Qwen/Qwen3-0.6B")) == {"GateProj", "Key", "Query"}


@pytest.mark.parametrize("placement", ["attn", "mlp", "all", "custom:Value,OutProj,UpProj"])
def test_adapter_count_matches_counter(placement):
    types = resolve_types(placement)
    model, n = build_lora_model(tiny(), types, rank=4)  # asserts equality internally
    assert n > 0
    assert target_modules(("Query",)) == ["q_proj"]


def test_only_chosen_types_get_adapters():
    model, _ = build_lora_model(tiny(), ("Value", "OutProj"), rank=2)
    names = [n for n, _ in model.named_parameters() if "lora_A" in n]
    assert names and all(("v_proj" in n or "o_proj" in n) for n in names)
    assert model.score is not None
    x = torch.randint(1, 100, (2, 5))
    assert model(input_ids=x).logits.shape == (2, 3)
