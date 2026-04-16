#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -f venv/bin/activate ]]; then
  source venv/bin/activate
fi
export CONS_EVAL_WORKERS="${CONS_EVAL_WORKERS:-6}"
WINDOWS=( ${EVAL_WINDOWS:-1 3 5 10} )
GROUPS_COUNT="${GROUPS_COUNT:-1000}"
if [[ -n "${POP_BIAS+x}" && -n "${POP_BIAS}" ]]; then
  POPULATION_BIASES="$POP_BIAS"
else
  POPULATION_BIASES="${POPULATION_BIASES:-0 1 2}"
fi
read -r -a BIAS_ARR <<< "$POPULATION_BIASES"
NDCG_K="${NDCG_K:-20}"
MODE="${MODE:-compute}"
PYTHON="${PYTHON:-python3}"
BASE_EXTRA=(
  --group-types similar outlier random
)
if [[ "${SKIP_DATASET:-0}" == "1" ]]; then
  :
else
  "$PYTHON" -m evaluation_frameworks.consensus_evaluation.evaluation.evaluation_preparation.eval_dataset_preparation \
    --group-size "${DATASET_GROUP_SIZE:-3}" \
    --min-com "${MIN_COM:-10}"
fi
EVAL_MODULES=(
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_with_sigmoid_policy_simple_priority_individual_rec
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_individual_rec
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_group_rec
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_without_feedback
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_with_feedback_ema
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_general_rec_individual
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_updatable
)
for W in "${WINDOWS[@]}"; do
  COMMON=(
    --mode "$MODE"
    --window-size "$W"
    --groups-count "$GROUPS_COUNT"
    --population-biases "${BIAS_ARR[@]}"
    --ndcg-k "$NDCG_K"
    "${BASE_EXTRA[@]}"
  )
  for mod in "${EVAL_MODULES[@]}"; do
    "$PYTHON" -m "$mod" "${COMMON[@]}"
  done
done
