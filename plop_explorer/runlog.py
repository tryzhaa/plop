"""Write result JSON stamped with what produced it, so committed numbers are checkable."""

import datetime
import json
import subprocess
from pathlib import Path

import torch
import transformers


def _git_hash():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=Path(__file__).parent
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_result(path, payload, **meta):
    """Write {"meta": {...}, "result": payload} to `path`; `meta` takes e.g. model, task, seed, dtype, device."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "git_hash": _git_hash(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        **meta,
    }
    path.write_text(json.dumps({"meta": meta, "result": payload}, indent=1, default=str))
    return path
