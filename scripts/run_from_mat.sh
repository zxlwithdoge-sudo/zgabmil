#!/bin/bash -l

#$ -N abmil_oct_full
#$ -j y
#$ -l h_rt=48:00:00
#$ -l gpu=1
#$ -pe omp 8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/work}"
RESUME="${RESUME:-0}"
WEIGHTS_PATH="${WEIGHTS_PATH:-}"

if [[ -z "${FRAMES_DIR:-}" ]]; then
  echo "FRAMES_DIR is required and must contain the flat .mat dataset."
  exit 2
fi
if [[ ! -d "$FRAMES_DIR" ]]; then
  echo "MAT frame directory does not exist: $FRAMES_DIR"
  exit 2
fi

DATA_ROOT="$WORK_DIR/data"
PATCHES_DIR="$DATA_ROOT/patches_112_masked"
FEATURES_DIR="$DATA_ROOT/features_112_masked"
SPLITS_DIR="$WORK_DIR/splits"
RESULTS_ROOT="$WORK_DIR/results"

if [[ "$RESUME" != "1" ]]; then
  for output_dir in \
    "$PATCHES_DIR" \
    "$FEATURES_DIR" \
    "$SPLITS_DIR" \
    "$RESULTS_ROOT"; do
    if [[ -d "$output_dir" ]] && [[ -n "$(ls -A "$output_dir")" ]]; then
      echo "Refusing to mix existing outputs in: $output_dir"
      echo "Use a new WORK_DIR, or set RESUME=1 for the same dataset."
      exit 2
    fi
  done
fi

mkdir -p "$DATA_ROOT" "$SPLITS_DIR" "$RESULTS_ROOT"
cd "$REPO_ROOT"

PREPROCESS_ARGS=(
  --frames-dir "$FRAMES_DIR"
  --patches-dir "$PATCHES_DIR"
  --patch-size 112
  --median-size 15
  --otsu-scale 0.35
  --min-tissue-fraction 0.2
  --min-component-size 1024
)
FEATURE_ARGS=(
  --patches-dir "$PATCHES_DIR"
  --features-dir "$FEATURES_DIR"
  --batch-size 128
)
if [[ "$RESUME" == "1" ]]; then
  PREPROCESS_ARGS+=(--skip-existing)
  FEATURE_ARGS+=(--skip-existing)
fi
if [[ -n "$WEIGHTS_PATH" ]]; then
  FEATURE_ARGS+=(--weights-path "$WEIGHTS_PATH")
fi

"$PYTHON_BIN" -m abmil_oct.preprocess "${PREPROCESS_ARGS[@]}"
"$PYTHON_BIN" -m abmil_oct.features "${FEATURE_ARGS[@]}"

DATA_ROOT="$DATA_ROOT" \
SPLITS_PATH="$SPLITS_DIR/AD_NC_folds.json" \
RESULTS_DIR="$RESULTS_ROOT/ad_nc" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/train_ad_nc.sh

DATA_ROOT="$DATA_ROOT" \
SPLITS_PATH="$SPLITS_DIR/CTE_NC_folds.json" \
RESULTS_DIR="$RESULTS_ROOT/cte_nc" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/train_cte_nc.sh

echo "Complete ABMIL-OCT outputs: $WORK_DIR"
