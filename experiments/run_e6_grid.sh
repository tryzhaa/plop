#!/usr/bin/env bash
# usage: STEPS=5000 SEEDS="0 1 2" bash experiments/run_e6_grid.sh <model> [placements...]
set -euo pipefail
MODEL="${1:?model}"; shift || true
PLACEMENTS=("$@"); [ ${#PLACEMENTS[@]} -eq 0 ] && PLACEMENTS=(plop mlp attn plop_inv)
for placement in "${PLACEMENTS[@]}"; do
  for seed in ${SEEDS:-0 1 2}; do
    RANK_ARG=""; [ "$placement" = "mlp" ] && RANK_ARG="--rank 8"   # the paper's MLP rank; the others are matched to its budget
    python experiments/e6_anli_train.py --model "$MODEL" --placement "$placement" --seed "$seed" \
      --steps "${STEPS:-5000}" $RANK_ARG
  done
done
