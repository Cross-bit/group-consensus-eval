#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/venv/bin/activate" ]]; then
  # Reuse one shared environment; do not copy virtualenv into job workspaces.
  source "$ROOT/venv/bin/activate"
fi

PYTHON="${PYTHON:-python3}"
MODE="${MODE:-compute}"
NDCG_K="${NDCG_K:-20}"
GROUPS_COUNT="${GROUPS_COUNT:-1000}"
GROUP_TYPES="${GROUP_TYPES:-similar outlier random}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/analysis_runs}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
WORKERS_PER_JOB="${WORKERS_PER_JOB:-20}"
SEED_WORKERS="${SEED_WORKERS:-32}"
SEED_FIRST="${SEED_FIRST:-1}"

WINDOWS=( ${EVAL_WINDOWS:-1 3 5 10} )
if [[ -n "${POP_BIAS+x}" && -n "${POP_BIAS}" ]]; then
  POPULATION_BIASES="$POP_BIAS"
else
  POPULATION_BIASES="${POPULATION_BIASES:-0 1 2}"
fi
read -r -a BIAS_ARR <<< "$POPULATION_BIASES"
read -r -a GROUP_TYPE_ARR <<< "$GROUP_TYPES"

# Keep BLAS/OpenMP from oversubscribing across many parallel jobs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

EVAL_MODULES=(
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_with_sigmoid_policy_simple_priority_individual_rec
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_individual_rec
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_group_rec
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_without_feedback
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_with_feedback_ema
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_general_rec_individual
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_updatable
)

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="$RUNS_ROOT/full_eval_parallel_$STAMP"
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"

echo "[parallel-eval] ROOT=$ROOT"
echo "[parallel-eval] BASE_DIR=$BASE_DIR"
echo "[parallel-eval] WINDOWS=${WINDOWS[*]}"
echo "[parallel-eval] BIASES=${BIAS_ARR[*]}"
echo "[parallel-eval] GROUP_TYPES=${GROUP_TYPE_ARR[*]}"
echo "[parallel-eval] PARALLEL_JOBS=$PARALLEL_JOBS WORKERS_PER_JOB=$WORKERS_PER_JOB"

if [[ "${SKIP_DATASET:-0}" != "1" ]]; then
  echo "[parallel-eval] Running dataset preparation once..."
  "$PYTHON" -m evaluation_frameworks.consensus_evaluation.evaluation.evaluation_preparation.eval_dataset_preparation \
    --group-size "${DATASET_GROUP_SIZE:-3}" \
    --min-com "${MIN_COM:-10}"
fi

if [[ "$SEED_FIRST" == "1" ]]; then
  echo "[parallel-eval] Seed pass: building shared heavy caches once (single writer)."
  SEED_W="${WINDOWS[0]}"
  SEED_B="${BIAS_ARR[0]}"
  SEED_G="${GROUP_TYPE_ARR[0]}"
  CONS_EVAL_WORKERS="$SEED_WORKERS" \
  "$PYTHON" -m "${EVAL_MODULES[0]}" \
    --mode "$MODE" \
    --window-size "$SEED_W" \
    --groups-count "$GROUPS_COUNT" \
    --population-biases "$SEED_B" \
    --ndcg-k "$NDCG_K" \
    --group-types "$SEED_G" \
    > "$LOG_DIR/seed.log" 2>&1 || true
  echo "[parallel-eval] Seed log: $LOG_DIR/seed.log"
fi

start_job() {
  local module="$1"
  local window="$2"
  local module_short="${module##*.}"
  local workdir="$BASE_DIR/work_w${window}_${module_short}"
  local logfile="$LOG_DIR/w${window}_${module_short}.log"

  mkdir -p "$workdir"

  # Full isolation per job: own cache/, own outputs, no cross-write race on pickle files.
  rsync -a --delete \
    --exclude '.git/' \
    --exclude 'venv/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude 'logs/' \
    --exclude '*.log' \
    "$ROOT/" "$workdir/"

  # Some modules resolve cached pickle paths via ".../analysis/cache/...".
  # Keep backward-compatible path shape inside isolated workdirs.
  ln -sfn . "$workdir/analysis"

  (
    cd "$workdir"
    CONS_EVAL_WORKERS="$WORKERS_PER_JOB" \
    "$PYTHON" -m "$module" \
      --mode "$MODE" \
      --window-size "$window" \
      --groups-count "$GROUPS_COUNT" \
      --population-biases "${BIAS_ARR[@]}" \
      --ndcg-k "$NDCG_K" \
      --group-types "${GROUP_TYPE_ARR[@]}"
  ) > "$logfile" 2>&1 &

  echo "[parallel-eval] started: w=$window module=$module_short pid=$! log=$logfile"
}

running=0
for w in "${WINDOWS[@]}"; do
  for mod in "${EVAL_MODULES[@]}"; do
    start_job "$mod" "$w"
    running=$((running + 1))
    if (( running >= PARALLEL_JOBS )); then
      wait -n
      running=$((running - 1))
    fi
  done
done

wait
echo "[parallel-eval] all jobs finished."
echo "[parallel-eval] logs: $LOG_DIR"

