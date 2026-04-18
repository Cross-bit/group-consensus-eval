# ====================================
# DESCRIPTION
# ====================================
# Analysis execution.
#
#

# =============================
# GROUP ALGORITHMS
# =============================
groups-generator:
	python3 -m evaluation_frameworks.consensus_evaluation.synthetic_groups.groups_generator

# =============================
# TUNNING
# =============================

MODE ?= auto # determines whether to recompute the evaluation options: auto, compute, load
# W: consensus window (--window-size); not related to dataset target below.
W ?= 10

# Run this script to generate the test dataset.
GROUP_SIZE ?= 10
# GROUPS_COUNT: how many groups each eval runs through the simulator (not users-per-group; that is --group-size for large-group targets).
GROUPS_COUNT ?= 1000
MIN_COM ?= 3

consensus-eval-dataset-gen:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluation_preparation.eval_dataset_preparation --group-size $(GROUPS_COUNT) --min-com $(MIN_COM)

tune_async_with_sigmoid_policy_simple_priority_individual_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_async_with_sigmoid_policy_simple_priority_individual_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_async_with_sigmoid_policy_simple_priority_group_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_async_with_sigmoid_policy_simple_priority_group_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_async_with_static_policy_simple_priority_individual_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_async_static_policy_simple_priority_function_individual_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_async_with_static_policy_simple_priority_group_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_async_static_policy_simple_priority_function_group_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_sync_without_feedback:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_sync_without_feedback --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_sync_with_feedback_ema:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_sync_with_feedback_ema --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_sync_with_feedback_mean:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_sync_with_feedback_mean --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_sync_with_feedback_mean_:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_sync_with_feedback_mean_ --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_hybrid_general_rec_individual:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_hybrid_general_rec_individual --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_hybrid: tune_hybrid_general_rec_individual
	@:

tune_hybrid_all_params:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_hybrid_all_params --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_hybrid_updatable:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_hybrid_individual_updatable --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

tune_hybrid_group_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.tune_hybrid_group_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

# =============================
# EVALUATIONS
# =============================

eval_async_with_sigmoid_policy_simple_priority_individual_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_with_sigmoid_policy_simple_priority_individual_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

eval_async_static_policy_simple_priority_function_individual_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_individual_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

eval_async_static_policy_simple_priority_function_group_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_async_static_policy_simple_priority_function_group_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

eval_sync_with_feedback_ema:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_with_feedback_ema --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

eval_sync_without_feedback:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_sync_without_feedback --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

eval_hybrid_general_rec_individual:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_general_rec_individual --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

eval_hybrid_updatable:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_updatable --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

# Alias na eval_hybrid_general_rec_individual (sweep first_round_ration → tune_hybrid_general_rec_individual)
eval_hybrid_general_rec_individual_by_first_round_ration:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_general_rec_individual_by_first_round_ration --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT)

# =============================
# LARGER GROUP EVALUATIONS
# =============================

eval_large_hybrid_group_updatable:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.larger_group_evaluations.eval_large_hybrid_group_updatable --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT) --group-size $(GROUP_SIZE)

eval_large_sync_with_feedback_ema:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.larger_group_evaluations.eval_large_sync_with_feedback_ema --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT) --group-size $(GROUP_SIZE)

eval_large_async_with_sigmoid_policy_simple_priority_group_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.larger_group_evaluations.eval_async_with_sigmoid_policy_simple_priority_group_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT) --group-size $(GROUP_SIZE)

eval_large_async_with_sigmoid_policy_simple_priority_individual_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.larger_group_evaluations.eval_async_with_sigmoid_policy_simple_priority_individual_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT) --group-size $(GROUP_SIZE)

# =============================
# EVALUATION SUMMARY
# =============================

table_rfc_comparision:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_comparisions --window-size $(W)

table_rfc_by_population_mood:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_by_population_mood --window-size $(W)

WINDOWS ?= 1 3 5 10
BIASES ?= 0 1 2
table_rfc_by_population_mood_all_windows:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_by_population_mood_all_windows --windows $(WINDOWS) --biases $(BIASES) --groups-count $(GROUPS_COUNT)

table_success_matches_all_windows:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_success_matches_all_windows --windows $(WINDOWS) --biases $(BIASES) --groups-count $(GROUPS_COUNT)

K ?= 10
table_ndcg_comparisions:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_ndcg_comparisions --k $(K) --window-size $(W)

table_unmatched_comparisions:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_unmatched_comparisions --window-size $(W) --bias 0

migrate-eval-cache-layout:
	python3 migrate_eval_cache_layout.py --window-size $(W) --eval-type test

table_rfc_large_group_size_comparisions:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.table_rfc_large_group_size_comparisions --window-size $(W) --groups-count $(GROUPS_COUNT) --group-sizes 3 5 7 10