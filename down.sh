#!/usr/bin/env bash
set -euo pipefail

REMOTE="cross-bit@groups-eval:/home/cross-bit/analysis/cache/cons_evaluations"
LOCAL_BASE="./cache/cons_evaluations"

ALGORITHMS=(
  "eval_async_static_policy_simple_priority_function_group_rec.py"
  "eval_async_static_policy_simple_priority_function_individual_rec.py"
  "eval_async_with_sigmoid_policy_simple_priority_individual_rec.py"
  "eval_large_hybrid_general_rec_individual.py"
  "eval_large_hybrid_group_updatable.py"
)

for W in 1 3 5 10; do
  for G in 5 7 10; do
    for ALGO in "${ALGORITHMS[@]}"; do
      local_dir="${LOCAL_BASE}/w_${W}/group_${G}/split_test/eval_n_1000/${ALGO}"
      mkdir -p "$local_dir"

      # Najdi další číslo: max(existing *.pkl) + 1, jinak 1
      max_num=0
      shopt -s nullglob
      for f in "$local_dir"/*.pkl; do
        bn="$(basename "$f" .pkl)"
        if [[ "$bn" =~ ^[0-9]+$ ]] && (( bn > max_num )); then
          max_num="$bn"
        fi
      done
      shopt -u nullglob
      next_num=$((max_num + 1))

      remote_file="${REMOTE}/w_${W}/group_${G}/split_test/eval_n_1000/${ALGO}/1.pkl"
      local_file="${local_dir}/${next_num}.pkl"

      echo "Downloading ${remote_file} -> ${local_file}"
      scp "$remote_file" "$local_file"
    done
  done
done