#!/bin/bash -l

#$ -N gabmil_oct_ablation
#$ -j y
#$ -l h_rt=24:00:00
#$ -l gpu=1
#$ -pe omp 8

# Trains the ABMIL baseline and the GABMIL spatial variants on one task, so the
# spatial information mixing module can be compared against plain ABMIL under
# identical splits, seeds, and hyperparameters.
#
#   TASK=ad_nc DATA_ROOT=/path/to/data SPLITS_PATH=/path/to/AD_NC_folds.pkl \
#   bash scripts/train_gabmil_ablation.sh
#
# MODES and WINDOW_SIZES may be overridden, for example:
#   MODES="none block" WINDOW_SIZES="2 3" bash scripts/train_gabmil_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-ad_nc}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/${TASK}_gabmil}"
MODES="${MODES:-none block grid block_grid}"
WINDOW_SIZES="${WINDOW_SIZES:-2 3 4}"
PATCH_SIZE="${PATCH_SIZE:-112}"

case "$TASK" in
  ad_nc)
    TASK_SCRIPT="train_ad_nc.sh"
    DEFAULT_SPLITS="$REPO_ROOT/splits/AD_NC_folds.json"
    ;;
  cte_nc)
    TASK_SCRIPT="train_cte_nc.sh"
    DEFAULT_SPLITS="$REPO_ROOT/splits/CTE_NC_folds.json"
    ;;
  *)
    echo "TASK must be ad_nc or cte_nc, got: $TASK"
    exit 2
    ;;
esac
SPLITS_PATH="${SPLITS_PATH:-$DEFAULT_SPLITS}"

cd "$REPO_ROOT"
mkdir -p "$RESULTS_ROOT"

run_variant() {
  local mode="$1"
  local size="$2"
  local name="$mode"
  if [[ "$mode" != "none" ]]; then
    name="${mode}_${size}"
  fi
  local results_dir="$RESULTS_ROOT/$name"
  if [[ -d "$results_dir" ]] && [[ -n "$(ls -A "$results_dir" 2>/dev/null)" ]]; then
    echo "Skipping existing variant: $results_dir"
    return
  fi
  echo "=== $TASK: spatial_mode=$mode window=$size ==="
  DATA_ROOT="$DATA_ROOT" \
  SPLITS_PATH="$SPLITS_PATH" \
  RESULTS_DIR="$results_dir" \
  PYTHON_BIN="$PYTHON_BIN" \
  SPATIAL_MODE="$mode" \
  BLOCK_SIZE="$size" \
  GRID_SIZE="$size" \
  PATCH_SIZE="$PATCH_SIZE" \
  bash "scripts/$TASK_SCRIPT"
}

for mode in $MODES; do
  if [[ "$mode" == "none" ]]; then
    run_variant none 1
  else
    for size in $WINDOW_SIZES; do
      run_variant "$mode" "$size"
    done
  fi
done

echo "Variant results under: $RESULTS_ROOT"
for summary in "$RESULTS_ROOT"/*/summary.json; do
  [[ -e "$summary" ]] || continue
  "$PYTHON_BIN" - "$summary" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text())
config = summary["config"]
variant = config.get("spatial_mode", "none")
if variant != "none":
    variant = f"{variant} (block={config['block_size']}, grid={config['grid_size']})"
print(
    f"{path.parent.name:16s} {variant:34s} "
    f"frame acc {summary['mean_test_accuracy']:.3f} +/- {summary['std_test_accuracy']:.3f}  "
    f"frame AUC {summary['mean_test_auc']:.3f} +/- {summary['std_test_auc']:.3f}"
)
PY
done
