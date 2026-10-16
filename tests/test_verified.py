"""The README's reproduction claims, checked against the committed summaries in reference/ours/.

Each summary is written by scripts/check_selection.py; a missing file skips its case.
"""

import json
from pathlib import Path

import pytest

OURS = Path(__file__).resolve().parents[1] / "reference" / "ours"
ALL_TASKS = ("math", "code", "history", "logic")

# (file stem, tasks that must match, paper PLoP set, paper PLoP^-1 set or None)
CLAIMS = [
    ("Llama-3.2-1B-Instruct", ALL_TASKS, {"Value", "OutProj", "DownProj"}, None),
    ("Qwen2.5-0.5B", ALL_TASKS, {"Value", "OutProj", "UpProj"}, None),
    ("Qwen3-0.6B", ALL_TASKS, {"DownProj", "UpProj", "Value"}, {"GateProj", "Key", "Query"}),
    # Out, Up and Down are a near-tie (0.95-0.99); the paper's picks come from math.
    ("Qwen3-1.7B", ("math",), {"DownProj", "OutProj", "Value"}, {"GateProj", "Key", "Query"}),
    # The paper's Gemma is the instruct model (-pt misses PLoP^-1 on every task). On code, Key and
    # Query are a near-tie (1.30 vs 1.28), so the reference setup picks Query there.
    ("gemma-3-1b-it", ("math", "history", "logic"), {"Key", "Value", "UpProj"}, {"OutProj", "GateProj", "DownProj"}),
]
CASES = [(stem, task, low, high) for stem, tasks, low, high in CLAIMS for task in tasks]


@pytest.mark.parametrize("stem,task,low,high", CASES, ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_selection_matches_paper(stem, task, low, high):
    path = OURS / f"{stem}_{task}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not generated")
    ref = json.loads(path.read_text())["result"]["reference"]
    assert set(ref["plop"]) == low
    if high:
        assert set(ref["plop_inv"]) == high


def test_fig4_layer_error_within_documented_gap():
    path = OURS / "fig4_layer_check.json"
    if not path.exists():
        pytest.skip("fig4_layer_check.json not generated")
    # Measured 3.1% per cell; fail if a change makes it clearly worse.
    assert json.loads(path.read_text())["result"]["mean_rel_err"]["reference"] < 0.04
