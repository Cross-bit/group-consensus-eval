#!/usr/bin/env bash
# Dvoufázová evaluace consensus skriptů (bash; spouštěj z kořene repa).
#
# Fáze 1 — jen async A0–A2 a sync S0–S1:
#   • okno W = PHASE1_W (výchozí 5; pro W=7: PHASE1_W=7 ./run_eval_phased_window.sh)
#   • --groups-count GROUPS_COUNT (výchozí 100)
#   • jen group types: similar, outlier, random
#
# Fáze 2 — až po doběhnutí fáze 1:
#   • všech 7 hlavních eval modulů (včetně hybridů H0, H1)
#   • W = PHASE2_W (výchozí 10), stejný GROUPS_COUNT, výchozí group types (5)
#
# Příklady:
#   chmod +x run_eval_phased_window.sh
#   ./run_eval_phased_window.sh
#   PHASE1_W=7 MODE=compute ./run_eval_phased_window.sh
#   PHASE1_W=1 PHASE2_W=1 ./run_eval_phased_window.sh   # obě fáze W=1 (jinak fáze 2 zůstane W=10)
#   PYTHON=python3 GROUPS_COUNT=200 ./run_eval_phased_window.sh
#   POPULATION_BIASES="0 0.5 1" ./run_eval_phased_window.sh   # výchozí je jen 0
#
# Paralelně s tune na W=10: cache je pod w_<W>/… — nekoliduje s w_10 tune.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
MODE="${MODE:-compute}"
GROUPS_COUNT="${GROUPS_COUNT:-100}"
PHASE1_W="${PHASE1_W:-5}"
PHASE2_W="${PHASE2_W:-10}"
# Jen unbiased (0); více hodnot: POPULATION_BIASES="0 0.5 1" …
POPULATION_BIASES="${POPULATION_BIASES:-0}"

# Fáze 1: A0, A1, A2, S0, S1 (bez hybridů)
PHASE1_MODULES=(
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_group_rec
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_individual_rec
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_with_sigmoid_policy_simple_priority_individual_rec
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_without_feedback
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_with_feedback_ema
)

# Fáze 2: stejných 5 + oba hybridy (7 modulů, pořadí jako run_consensus_eval_fast.sh)
PHASE2_MODULES=(
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_with_sigmoid_policy_simple_priority_individual_rec
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_individual_rec
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_group_rec
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_without_feedback
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_with_feedback_ema
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_general_rec_individual
  evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_updatable
)

run_mod_phase1() {
  local mod="$1"
  echo "###########################################################################"
  echo "###  Phase 1 (A0–A2, S0–S1)  W=${PHASE1_W}  groups=${GROUPS_COUNT}  biases=${POPULATION_BIASES}  ###"
  echo "###  $mod"
  echo "###########################################################################"
  "$PYTHON" -m "$mod" \
    --mode "$MODE" \
    --window-size "$PHASE1_W" \
    --groups-count "$GROUPS_COUNT" \
    --population-biases $POPULATION_BIASES \
    --group-types similar outlier random
}

run_mod_phase2() {
  local mod="$1"
  echo "###########################################################################"
  echo "###  Phase 2 (all 7)  W=${PHASE2_W}  groups=${GROUPS_COUNT}  biases=${POPULATION_BIASES}  ###"
  echo "###  $mod"
  echo "###########################################################################"
  "$PYTHON" -m "$mod" \
    --mode "$MODE" \
    --window-size "$PHASE2_W" \
    --groups-count "$GROUPS_COUNT" \
    --population-biases $POPULATION_BIASES
}

echo "== Phase 1: ${#PHASE1_MODULES[@]} modulů, W=${PHASE1_W}, group types = similar outlier random, groups-count=${GROUPS_COUNT}, population-biases=${POPULATION_BIASES} =="
for mod in "${PHASE1_MODULES[@]}"; do
  run_mod_phase1 "$mod"
done

echo ""
echo "== Phase 1 hotová. Spouštím Phase 2: ${#PHASE2_MODULES[@]} modulů, W=${PHASE2_W}, výchozí group types, groups-count=${GROUPS_COUNT}, population-biases=${POPULATION_BIASES} =="

for mod in "${PHASE2_MODULES[@]}"; do
  run_mod_phase2 "$mod"
done

echo ""
echo "== Obě fáze hotové."
echo "    Cache: cache/cons_evaluations/w_${PHASE1_W}/… (fáze 1), w_${PHASE2_W}/… (fáze 2)"
