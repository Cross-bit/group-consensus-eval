#!/usr/bin/env bash
set -euo pipefail

SRC="${SRC:-$HOME/analysis}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/analysis_runs}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# Prevent BLAS/OpenMP oversubscription when running many workers.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

W1_WORKERS="${W1_WORKERS:-45}"
W3_WORKERS="${W3_WORKERS:-45}"
GROUPS_COUNT="${GROUPS_COUNT:-1000}"
POPULATION_BIASES="${POPULATION_BIASES:-0 1 2}"

mkdir -p "$RUNS_ROOT/logs"

W1_DIR="$RUNS_ROOT/w1_${STAMP}"
W3_DIR="$RUNS_ROOT/w3_${STAMP}"

echo "[prep] Creating isolated workspaces..."
mkdir -p "$W1_DIR" "$W3_DIR"
rsync -a --delete \
  --exclude '.git/' \
  --exclude 'venv/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  "$SRC/" "$W1_DIR/"
rsync -a --delete \
  --exclude '.git/' \
  --exclude 'venv/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  "$SRC/" "$W3_DIR/"

echo "[run] Starting W=1 and W=3 in parallel (isolated caches)..."
(
  cd "$W1_DIR"
  CONS_EVAL_WORKERS="$W1_WORKERS" GROUPS_COUNT="$GROUPS_COUNT" EVAL_WINDOWS="1" POPULATION_BIASES="$POPULATION_BIASES" RUN_NUM=1001 \
    ./run_eval_full_biases_windows.sh
) > "$RUNS_ROOT/logs/w1_${STAMP}.log" 2>&1 &
PID1=$!

(
  cd "$W3_DIR"
  CONS_EVAL_WORKERS="$W3_WORKERS" GROUPS_COUNT="$GROUPS_COUNT" EVAL_WINDOWS="3" POPULATION_BIASES="$POPULATION_BIASES" RUN_NUM=1003 \
    ./run_eval_full_biases_windows.sh
) > "$RUNS_ROOT/logs/w3_${STAMP}.log" 2>&1 &
PID2=$!

wait "$PID1"
wait "$PID2"

echo "[done] All jobs finished."
echo "[done] Logs:"
echo "  $RUNS_ROOT/logs/w1_${STAMP}.log"
echo "  $RUNS_ROOT/logs/w3_${STAMP}.log"

