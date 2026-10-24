import math

import pytest

from plop_explorer import data
from plop_explorer.stats import mean_se, pick, pick_change_rate, spearman


def test_spearman_orders():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)


def test_pick_and_change_rate():
    a = {"Q": 2, "K": 3, "V": 0.9, "O": 1.0, "D": 1.1}
    b = {**a, "D": 5, "K": 1.05}
    assert pick(a) == {"V", "O", "D"} and pick(b) == {"V", "O", "K"}
    assert pick_change_rate([a, a]) == 0 and pick_change_rate([a, b]) == 1
    assert pick_change_rate([a, a, b]) == pytest.approx(2 / 3)


def test_mean_se():
    m, se = mean_se([1.0, 2.0, 3.0])
    assert m == 2.0 and se == pytest.approx(1 / math.sqrt(3))


def test_shuffled_words_keeps_words(monkeypatch):
    monkeypatch.setattr(data, "load_prompts", lambda task, n, seed: ["a b c d e f", "x y z"])
    out = data.load_control("shuffled_words", seed=1)
    assert [sorted(o.split()) for o in out] == [sorted("a b c d e f".split()), ["x", "y", "z"]]


def test_random_tokens_same_length_no_specials(monkeypatch):
    class Tok:
        all_special_ids = [0]

        def __len__(self):
            return 50

        def __call__(self, text):
            return {"input_ids": text.split()}

        def decode(self, ids):
            assert 0 not in ids
            return " ".join(map(str, ids))

    monkeypatch.setattr(data, "load_prompts", lambda task, n, seed: ["a b c", "d e"])
    out = data.load_control("random_tokens", tokenizer=Tok())
    assert [len(o.split()) for o in out] == [3, 2]
