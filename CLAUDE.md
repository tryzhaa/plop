# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reproduction and explorer for the PLoP paper (Hayou, Ghosh, Yu, arXiv:2506.20629): score every linear module type in an LLM by Normalized Feature Norm (NFN) on a task's prompts, then recommend the lowest-scoring types for LoRA placement. `docs/plop-implementation-plan.md` is the source of truth for status, gates (G1-G9), experiment IDs (E*/X*), and product decisions; it overrides `docs/plop-strategy.md` where they differ. Read its Section 0 before starting new work. The planned FastAPI + Plotly dashboard is not built yet.

## Commands

```bash
pip install -r requirements.txt            # torch, transformers>=5, datasets, pytest
pytest                                     # all unit tests
pytest tests/test_nfn.py::test_name        # single test
python scripts/check_reference.py --task math                          # Llama-1B vs paper's 7-number table
python scripts/check_selection.py --models Qwen/Qwen3-0.6B --tasks math # PLoP picks vs paper's sets
```

The scripts download models and datasets from Hugging Face and default to `mps` if available, else CPU. Gemma overflows in fp16, so `check_selection.py` retries in fp32. Outputs go to `results/` (gitignored).

## Architecture

- `plop_explorer/nfn.py`: `score_model(model, batches, mask_padding=True)` registers forward hooks on every `q/k/v/o/gate/up/down_proj` Linear and accumulates `||W z|| / ||z||` from the module's own output (no extra matmul), divided by the closed-form random gain `||W||_F / sqrt(d_in)`. This replaces the reference code's sampled Gaussian baseline. Only the transformer body runs (`model.base_model`), so the LM head is skipped. `NFNResult` gives `by_type()`, `grid()` (type x layer heatmap) and `recommend(k)`. Non-finite scores raise `FloatingPointError`.
- `plop_explorer/data.py`: seeded prompt sets for `math`, `code`, `history`, `logic`, built from the same sources and formatting as the authors' code. `make_batches(..., padding_side="right")` with an EOS pad token mimics the reference setup.
- `plop_explorer/lora_params.py`: LoRA parameter counts per placement, plus `match_rank` for equal-budget comparisons. `PLACEMENTS["Attn"]` is Q, K, V only (no O), the only reading that reproduces the paper's ranks.
- `reference/plop_reference_by_type.json`: the authors' published per-type scores, used as the ground truth by `check_reference.py`. The JSON's `actual`/`random` scale is W scaled to unit RMS, divided by sqrt(d_in); `nfn.py` reports both that scale and plain `nfn`.
- `scripts/check_*.py`: validation gates, not library code. They insert the repo root on `sys.path` and import from `plop_explorer`.

## Conventions worth knowing

- Two scoring modes matter: `mask_padding=False` plus right-padding reproduces the authors' numbers (faithfulness mode); `mask_padding=True` is our default. Any change to `nfn.py` or `data.py` should keep `check_reference.py` within the documented gap (mean error about 2.5%, Llama-1B math top-3 matches).
- Module types are matched by leaf name (`q_proj`, etc.) and layer index by `.layers.N.`. Architectures with other names are unsupported and raise a `ValueError`.
- Per-type aggregation is an unweighted mean over layers, as in the paper.
