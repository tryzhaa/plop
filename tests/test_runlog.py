import json

from plop_explorer.runlog import save_result


def test_save_result_stamps_meta(tmp_path):
    p = save_result(tmp_path / "sub" / "r.json", {"Value": 0.97}, seed=0, dtype="float16")
    d = json.loads(p.read_text())
    assert d["result"] == {"Value": 0.97}
    assert {"git_hash", "torch", "transformers", "timestamp", "seed", "dtype"} <= d["meta"].keys()
