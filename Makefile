# ====================================
# DESCRIPTION
# ====================================
# Analysis execution.
#
#



# =============================
# RESTAURANT
# =============================

# Generates new dataset of restaurants.
restaurant-dataset-generate:
	python3 -m restaurant_data


#
# KNN
#
###

#
#restaurant-knni-regularisation: # NOTICE: irrelevant due to matrix sparsity... viz restaurant-knn-avg-neighbors
#	python3 -m restaurant_data.algo_experiments.optimal_item_item_k
#
#restaurant-knn-regularisation:
#	python3 -m restaurant_data.algo_experiments.optimal_knn_user_user_k

restaurant-knn-avg-neighbors-by-test-size:
	python3 -m restaurant_data.algo_experiments.knn_neighbors_count_by_test_size


#
# EASER
#
###

restaurant-easer-regularisation:
	python3 -m restaurant_data.algo_experiments.optimal_easer_lambda


#
# Math (sigmoid)
# ============================

math-sigmoid-upper-bounded:
	python3 -m math_analysis.sigmoid-offset-upper-bounded

math-sigmoid:
	python3 -m math_analysis.sigmoid-offset

math-sigmoid-eval-agent:
	python3 -m math_analysis.sigmoid-eval-agent

#
# General
#
###

# =============================
# RESTAURANT
# =============================

restaurant-general-stats:
	python3 -m restaurant_data.general_analysis

restaurant-places-coverage-people:
	python3 -m restaurant_data.places_coverage.cumulative_coverage_with_people

restaurant-places-coverage:
	python3 -m restaurant_data.places_coverage.cumulative_coverage

restaurant-reviews-histogram:
	python3 -m restaurant_data.places_coverage.users_ratings_frequencies

restaurant-algo-comparison:
	python3 -m restaurant_data.algo_experiments.comparison_of_algorithms

restaurant-knn-by-test-size:
	python3 -m restaurant_data.algo_experiments.knn_by_dataset_size

restaurant-knn-parameter-tuning:
	python3 -m restaurant_data.algo_experiments.knn_parameter_tuning

restaurant-sparsity-dataset:
	python3 -m restaurant_data.sparsity_issue.sparsity_users_elimination_method

restaurant-overlaps-by-test-size:
	python3 -m restaurant_data.sparsity_issue.overlaps_by_dataset_size

restaurant-cb-model-run:
	python3 -m restaurant_data.algo_experiments.algos.cb_model

restaurant-cb-evaluation:
	python3 -m restaurant_data.algo_experiments.cb_model_precision


# =============================
# MOVIES
# =============================

movie-dataset-stats:
	python3 -m movies_data.genera_stats

movie-dataset-genre-frequencies:
	python3 -m movies_data.genres_coverage_frequencies

movie-popularity-histogram:
	python3 -m movies_data.popularity_histogram

init-size:
	python3 -m movies_data.initialisation_size

movie-knn-algo-comparison:
	python3 -m movies_data.algo_experiments.comparison_of_algorithms

init-sampling-all:
	python3 -m movies_data.initialisation_sampling.init_sampling_test_all

movie-release-year-popularity:
	python3 -m movies_data.release_year_popularity_histogram

movie-dataset-stats:
	python3 -m movies_data.genera_stats

# Production app Movie lens dataset filtering

movie-prod-filter:
	python3 -m movies_data.production_filtering.filter_dataset --input-dir ./movies_data/dataset/ml-32m/ --output-dir ./movies_data/production_filtering

movie-distribution-year-release-vs-easer:
	python3 -m movies_data.distribution_year_release_vs_easer

# =============================
# CLUSTERING
# =============================

# Analysis of kmeans clustering over MovieLens 1M
kmeans-analysis:
	python3 -m movies_data.initialisation_sampling.clustering.kmeans_analysis

# =============================
# GROUP ALGORITHMS
# =============================
group-recommender:
	python3 -m evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender

consensus-mediator:
	python3 -m evaluation_frameworks.consensus_evaluation.consensus_mediator

groups-generator:
	python3 -m evaluation_frameworks.consensus_evaluation.synthetic_groups.groups_generator


## Group algorithm evaluation

easer_filtered_movielens_tuning:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.other_evaluations.easer_filtered_movielens_tuning


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

#
# evaluation
#

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

## larger groups:

eval_large_hybrid_group_updatable:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.larger_group_evaluations.eval_large_hybrid_group_updatable --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT) --group-size $(GROUP_SIZE)

tune_large_hybrid_group_updatable:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.larger_group_evaluations.tune_large_hybrid_group_updatable --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT) --group-size $(GROUP_SIZE)

eval_large_sync_with_feedback_ema:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.larger_group_evaluations.eval_large_sync_with_feedback_ema --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT) --group-size $(GROUP_SIZE)

eval_large_async_with_sigmoid_policy_simple_priority_group_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.larger_group_evaluations.eval_async_with_sigmoid_policy_simple_priority_group_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT) --group-size $(GROUP_SIZE)

eval_large_async_with_sigmoid_policy_simple_priority_individual_rec:
	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.evaluations.larger_group_evaluations.eval_async_with_sigmoid_policy_simple_priority_individual_rec --mode $(MODE) --window-size $(W) --groups-count $(GROUPS_COUNT) --group-size $(GROUP_SIZE)

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


#
# evaluation summary
#


#consensus-eval:
#	python3 -m evaluation_frameworks.consensus_evaluation.evaluation.eval_async

# =============================
# API & DATA COLLECTIONS
# =============================
google-places-api:
	python3 -m movies_data.
	python3 -m restaurant_data.dataset.scripts.google_places_api

tmdb-test-movie-data:
	python3 -m movies_data.tmdb_api.get_data_test

tmdb-filter-valid-rows-ids:
	python3 -m movies_data.tmdb_api.check_links_correct_mappings.remove_empty_ids --dedupe

tmdb-validate-ids:
	python3 -m movies_data.tmdb_api.check_links_correct_mappings.validate_tmdb_ids

tmdb-find-missing-ids:
	python3 -m movies_data.tmdb_api.check_links_correct_mappings.find_missing_tmdb_ids

tmdb-resolve-missing-ids:
	python3 -m movies_data.tmdb_api.check_links_correct_mappings.resolve_missing_tmdb_ids

tmdb-filter-repaired-ids:
	python3 -m movies_data.tmdb_api.check_links_correct_mappings.filter_repaired_ids