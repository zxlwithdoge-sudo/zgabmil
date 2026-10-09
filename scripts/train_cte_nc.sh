#!/bin/bash -l

#$ -N abmil_cte_nc
#$ -j y
#$ -l h_rt=06:00:00
#$ -l gpu=1
#$ -pe omp 8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
SPLITS_PATH="${SPLITS_PATH:-$REPO_ROOT/splits/CTE_NC_folds.json}"
SPATIAL_MODE="${SPATIAL_MODE:-none}"
BLOCK_SIZE="${BLOCK_SIZE:-3}"
GRID_SIZE="${GRID_SIZE:-3}"
PATCH_SIZE="${PATCH_SIZE:-112}"
FLUSH_DENORMALS="${FLUSH_DENORMALS:-0}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/cte_nc}"

cd "$REPO_ROOT"

if [[ ! -f "$SPLITS_PATH" ]]; then
  echo "Creating deterministic CTE/NC subject splits: $SPLITS_PATH"
  "$PYTHON_BIN" -m abmil_oct.create_splits \
    --features-dir "$DATA_ROOT/features_112_masked" \
    --label0 NC \
    --label1 CTE \
    --output "$SPLITS_PATH" \
    --folds 10 \
    --test-per-class 2 \
    --seed 0
fi

"$PYTHON_BIN" -m abmil_oct.train \
  --features-dir "$DATA_ROOT/features_112_masked" \
  --splits-path "$SPLITS_PATH" \
  --results-dir "$RESULTS_DIR" \
  --label0 NC \
  --label1 CTE \
  --folds 10 \
  --epochs 12 \
  --learning-rate 2e-4 \
  --weight-decay 1e-5 \
  --dropout 0.25 \
  --spatial-mode "$SPATIAL_MODE" \
  --block-size "$BLOCK_SIZE" \
  --grid-size "$GRID_SIZE" \
  --patch-size "$PATCH_SIZE" \
  --seed 0 \
  --num-workers 8 \
  ${FLUSH_DENORMALS:+$([[ "$FLUSH_DENORMALS" == "1" ]] && echo --flush-denormals)}

"$PYTHON_BIN" -m abmil_oct.evaluate \
  --results-dir "$RESULTS_DIR" \
  --features-dir "$DATA_ROOT/features_112_masked"
