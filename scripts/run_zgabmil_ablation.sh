#!/usr/bin/env bash
# GABMIL + z-axis feature mixing, 10 predefined subject folds.
#
# Task is chosen by TASK=ad_nc (default) or TASK=cte_nc, which sets the
# splits file, the positive label and the output directory together - the
# three must agree or training fails on the first subject.
#
# Same bag (one frame), same per-frame loss, same majority vote as the
# manuscript baseline, so every row lines up with the GABMIL ablation table.
# Only the SIMM changes: a third mixing stage along the frame axis.
#
#   z1 / z3 / z5     z mixing alone at three window sizes
#   block3_z3        in-plane BLOCK 3 plus z 3
#   grid3_z3         in-plane GRID 3 plus z 3
#
# Reference points from the existing runs:
#   ABMIL baseline        frame AUC 0.758 +/- 0.184, subject 0.725
#   BLOCK_3 (in-plane)    frame AUC 0.600            <- in-plane mixing hurt
#   z-smoothing w=5       frame AUC 0.780            <- inference only, no training
#
# Arms already holding a subject_summary.json are skipped; FORCE=1 redoes them.
set -euo pipefail

TASK="${TASK:-ad_nc}"
case "${TASK}" in
  ad_nc)  DEFAULT_SPLITS=abmil_oct/splits/AD_NC_folds.json;  DEFAULT_LABEL1=AD ;;
  cte_nc) DEFAULT_SPLITS=abmil_oct/splits/CTE_NC_folds.json; DEFAULT_LABEL1=CTE ;;
  *) echo "TASK must be ad_nc or cte_nc, got '${TASK}'" >&2; exit 1 ;;
esac

FEATURES="${FEATURES:-abmil_oct/data/features_112_masked}"
SPLITS="${SPLITS:-${DEFAULT_SPLITS}}"
LABEL0="${LABEL0:-NC}"
LABEL1="${LABEL1:-${DEFAULT_LABEL1}}"
OUT="${OUT:-results/${TASK}_zgabmil}"
EPOCHS="${EPOCHS:-12}"
PYTHON="${PYTHON:-python}"

run_arm() {
  local name=$1 mode=$2 zwin=$3 bsize=${4:-3} gsize=${5:-3}
  local dir="${OUT}/${name}"
  if [ -f "${dir}/subject_summary.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "=== skip ${name} (already done; FORCE=1 to redo)"
    return
  fi
  echo "=== ${TASK} ${name}: spatial-mode=${mode} z-window=${zwin}"
  "${PYTHON}" -m abmil_oct.train_zgabmil \
    --features-dir "${FEATURES}" --splits "${SPLITS}" \
    --label0 "${LABEL0}" --label1 "${LABEL1}" --results-dir "${dir}" \
    --spatial-mode "${mode}" --z-window "${zwin}" \
    --block-size "${bsize}" --grid-size "${gsize}" \
    --epochs "${EPOCHS}"
}

run_arm z1        none  1
run_arm z3        none  3
run_arm z5        none  5
run_arm block3_z3 block 3 3
run_arm grid3_z3  grid  3 3 3

echo
if [ "${TASK}" = "cte_nc" ]; then
  echo "summaries (CTE baseline: frame AUC 0.461 - BELOW CHANCE."
  echo " Both improvement and degradation move this metric toward 0.50,"
  echo " so no arm here supports a claim about the module.)"
else
  echo "summaries (baseline: frame AUC 0.758, subject 0.725):"
fi
for d in "${OUT}"/*/; do
  [ -f "${d}subject_summary.json" ] || continue
  printf '%-12s ' "$(basename "$d")"
  "${PYTHON}" - "${d}subject_summary.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
c=s["config"]
print(f"mode={c['spatial_mode']:<11s} z={c['z_window']}  "
      f"frame AUC {s['mean_frame_auc']:.3f} +/- {s['std_frame_auc']:.3f}   "
      f"balanced {s['mean_balanced_accuracy']:.3f}   "
      f"subject {s['mean_subject_accuracy']:.3f} +/- {s['std_subject_accuracy']:.3f}")
PY
done
