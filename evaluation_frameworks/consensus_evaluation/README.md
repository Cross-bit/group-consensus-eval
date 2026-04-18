# Consensus evaluation package

This package implements the **group consensus** experiment code used with MovieLens-style pipelines: mediators (async / sync / hybrid), round-based user feedback simulation, and evaluation scripts that produce metrics and LaTeX-friendly tables.

For **how to run** experiments (venv, dataset layout, `make consensus-*`, environment variables), see the repository root [README.md](../../README.md) and `./run_consensus.sh help`.

**Single experiments** (one `eval_*` / `tune_*` module, tables, larger-group scripts) are usually started from the repo root via **`Makefile`** targets; see the root README section *Makefile and `./run_consensus.sh`*.

The sibling package **`general_recommender_evaluation/`** holds separate MovieLens / Surprise tooling; it is not invoked by `run_consensus.sh`.

## Layout

| Path | Role |
|------|------|
| `consensus_mediator.py` | Mediator orchestration and threshold policies (package entry point used by evaluators). |
| `consensus_algorithm/` | Core algorithm pieces: `recommender_engine`, `redistribution_unit`, `priority_queue`, `models`. |
| `synthetic_groups/` | Synthetic group construction, test-set splits, embedding helpers, generator tests. |
| `evaluation/` | End-to-end evaluation: preparation, algo settings, and runnable `evaluations/` (tune scripts, larger-group variants, print/export). |
| `evaluation/evaluations/evaluators/` | Evaluator runners, consensus evaluator, and `consensus_evaluation_agents/` (simulated user agents). |

Nested READMEs (dataset paths, preparation steps):

- [evaluation/evaluation_preparation/README.md](evaluation/evaluation_preparation/README.md)
- [evaluation/algo_eval_settings/README.md](evaluation/algo_eval_settings/README.md)
