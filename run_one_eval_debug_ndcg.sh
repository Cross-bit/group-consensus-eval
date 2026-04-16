#!/usr/bin/env bash
set -euo pipefail

# One concrete example:
# - runs a single evaluation module
# - enables parallel workers
# - writes debug profile
# - prints NDCG table from the latest produced pickle
#
# Usage:
#   chmod +x run_one_eval_debug_ndcg.sh
#   ./run_one_eval_debug_ndcg.sh
# Optional overrides:
#   CONS_EVAL_WORKERS=6 GROUPS_COUNT=100 GROUP_TYPES="random outlier" ./run_one_eval_debug_ndcg.sh
#
# Reproduce old thesis-era context bug (train split + outlier groups, ignores CLI group_type):
#   CONS_EVAL_REINTRODUCE_OLD_CONTEXT_GROUPS_BUG=1 ./run_one_eval_debug_ndcg.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

# Example module (single script)
MODULE="${MODULE:-evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_individual_rec}"
EVAL_NAME="${EVAL_NAME:-eval_async_static_policy_simple_priority_function_individual_rec.py}"

# Fast debug/test defaults
W="${W:-10}"
GROUPS_COUNT="${GROUPS_COUNT:-100}"
POP_BIAS="${POP_BIAS:-0}"
NDCG_K="${NDCG_K:-20}"
GROUP_TYPES="${GROUP_TYPES:-random}"
MODE="${MODE:-compute}"
SAVE_MODE="${SAVE_MODE:-append}"  # append = always write a new numbered pickle
WORKERS="${CONS_EVAL_WORKERS:-6}"

export CONS_EVAL_WORKERS="$WORKERS"
export PYTHONPATH="${PYTHONPATH:-.}"

OLD_CONTEXT_BUG=0
_b="${CONS_EVAL_REINTRODUCE_OLD_CONTEXT_GROUPS_BUG:-}"
if [[ "${_b,,}" == "1" || "${_b,,}" == "true" || "${_b,,}" == "yes" ]]; then
  export CONS_EVAL_REINTRODUCE_OLD_CONTEXT_GROUPS_BUG
  OLD_CONTEXT_BUG=1
  echo "!! CONS_EVAL_REINTRODUCE_OLD_CONTEXT_GROUPS_BUG=$_b (old build_context_groups_data hardcode active)"
fi
unset _b

echo "== Run one evaluation =="
echo "module=$MODULE"
echo "workers=$CONS_EVAL_WORKERS, W=$W, groups_count=$GROUPS_COUNT, ndcg_k=$NDCG_K, group_types=[$GROUP_TYPES]"
echo "------------------------------------------------------------------------"
echo " POPULATION MOOD BIAS → předáno jako --population-biases: $POP_BIAS"
echo " (více hodnot v experimentu = více bloků >>> Bias k/n v Python výstupu)"
echo "------------------------------------------------------------------------"

PROFILE_TAG="workers_${CONS_EVAL_WORKERS}"
if [[ "$OLD_CONTEXT_BUG" -eq 1 ]]; then
  PROFILE_TAG="${PROFILE_TAG}_old_context_bug"
fi

"$PYTHON" -m "$MODULE" \
  --mode "$MODE" \
  --window-size "$W" \
  --groups-count "$GROUPS_COUNT" \
  --population-biases "$POP_BIAS" \
  --ndcg-k "$NDCG_K" \
  --group-types $GROUP_TYPES \
  --save-mode "$SAVE_MODE" \
  --debug-profile \
  --debug-profile-tag "$PROFILE_TAG"

echo ""
echo "== Resolve latest pickle path =="
LATEST_PKL="$(
W="$W" EVAL_NAME="$EVAL_NAME" GROUPS_COUNT="$GROUPS_COUNT" "$PYTHON" - <<'PY'
import os
from pathlib import Path
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import evaluation_results_dir

w = os.environ["W"]
eval_type = "test"
evaluation_name = os.environ["EVAL_NAME"]
groups_count = int(os.environ["GROUPS_COUNT"])

base = evaluation_results_dir(
    window_size=w,
    eval_type=eval_type,
    evaluation_name=evaluation_name,
    groups_count=groups_count,
    layout="labeled",
)

pickles = list(base.glob("*.pkl"))
if not pickles:
    raise SystemExit(f"No pickle found in: {base}")

latest = max(pickles, key=lambda p: p.stat().st_mtime)
print(str(latest))
PY
)"

echo "latest_pickle=$LATEST_PKL"
echo ""
echo "== NDCG table =="
"$PYTHON" inspect_ndcg_pickle.py --pickle "$LATEST_PKL" --k "$NDCG_K" --bias "$POP_BIAS"

