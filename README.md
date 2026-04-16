# Consensus Evaluation Framework (Paper Version)

This repository contains the experimental framework used for group consensus evaluation on MovieLens data, focused on:

- async mediators,
- sync mediators,
- hybrid mediators,
- RFC/NDCG-style evaluation outputs and LaTeX table generation.

The core implementation is in `evaluation_frameworks/consensus_evaluation`.

## Framework Overview

![Consensus Framework Diagram](docs/consensus_framework.png)

At a high level:

- a **General Recommender** provides candidate items,
- a **Redistribution Unit** adapts ranking according to group interaction state,
- users provide iterative feedback in rounds,
- evaluator scripts measure convergence quality (e.g., rounds to consensus, success ratio, NDCG diagnostics).

## Repository Layout

- `evaluation_frameworks/` — consensus algorithms, evaluators, context factories, print/export scripts
- `movies_data/` — dataset loading and sparse cache usage for MovieLens pipelines (see `movies_data/README.md`)
- `utils/` — shared config/cache helpers (including `load_or_build_pickle`)
- `cache/cons_evaluations/` — evaluation outputs used by reporting scripts
- `run_eval*.sh` — orchestration scripts for standard and parallel runs
- `unit_tests/` — targeted tests for mediator logic

## Quick Start

### 1) Environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### 1.1) Dataset note (important)

This repository does not track large MovieLens 32M raw/cache files in Git.

- Read `movies_data/README.md` for expected local files.
- Download the dataset locally and place it under `movies_data/dataset/ml-32m/` before running full evaluations.

### 2) One-shot full evaluation (single workspace)

```bash
CONS_EVAL_WORKERS=8 GROUPS_COUNT=1000 EVAL_WINDOWS="1 3 5 10" POPULATION_BIASES="0 1 2" ./run_eval_full_biases_windows.sh
```

### 3) High-throughput isolated parallel run (recommended on multi-core VM)

```bash
SEED_FIRST=1 RUNS_ROOT=/dev/shm/analysis_runs EVAL_WINDOWS="1 3" POPULATION_BIASES="0 1 2" GROUPS_COUNT=1000 PARALLEL_JOBS=5 WORKERS_PER_JOB=20 ./run_eval_full_biases_windows_parallel_seeded.sh
```

Why isolated runs matter:

- each job gets its own workspace/cache,
- parallel processes do not write to the same pickle path,
- avoids `pickle data was truncated` race-condition failures.

## Evaluation Flow

1. **Dataset preparation / context load**
   - sparse ratings + filtered dataset context
2. **Group set loading**
   - similar/outlier/random (+ optional divergent/variance variants)
3. **Recommendation model loading**
   - Easer/SVD loaded from cache or trained once and cached
4. **Round-based simulation**
   - feedback loop with mediator policy
5. **Metrics + persistence**
   - results persisted under `cache/cons_evaluations/...`
6. **Reporting**
   - export + LaTeX table scripts in `evaluation/evaluations/print/`

## Adding a New Evaluation Script

Create a new module under:

- `evaluation_frameworks/consensus_evaluation/evaluation/evaluations/`

Recommended pattern:

1. Define an experiment class compatible with existing evaluation modules.
2. Reuse `Runner`/context factory utilities instead of custom data-loading logic.
3. Expose CLI via existing `autorun(...)` pattern from `evaluations/config.py`.
4. Keep output keys consistent with current result schema so print/export scripts can consume it.
5. Add module entry to `run_eval_full_biases_windows.sh` (and optionally parallel orchestration script).

Minimal checklist for compatibility:

- accepts `--window-size`, `--groups-count`, `--population-biases`, `--group-types`, `--ndcg-k`
- saves results into the existing `cons_evaluations` layout
- does not write to globally shared ad-hoc cache paths

## Redistribuční jednotka (Redistribution Unit)

The redistribution unit is the adaptive block between recommendation candidates and user-visible ordering. It is responsible for:

- incorporating current round feedback,
- balancing exploitation vs. exploration under mediator policy,
- producing updated item ranking for the next interaction round.

Implementation references:

- `evaluation_frameworks/consensus_evaluation/consensus_algorithm/redistribution_unit.py`
- related mediator orchestration in `consensus_mediator.py`

## Reproducibility Notes

- Keep `POPULATION_BIASES`, `EVAL_WINDOWS`, `GROUPS_COUNT`, and `group-types` fixed across compared algorithms.
- Prefer one dedicated cache build before long runs.
- On large VMs, use isolated parallel orchestration script with `/dev/shm` if disk I/O or disk size is limiting.
- Do not run multiple non-isolated full eval scripts in the same workspace.
- Keep large local dataset/cache artifacts out of Git (already enforced by `.gitignore`).

## Common Issues

- **`pickle data was truncated`**  
  Usually caused by interrupted or concurrent write to shared pickle. Rebuild the affected file and use isolated parallel workspaces.

- **`No space left on device`**  
  Increase disk or run isolated workspaces under `/dev/shm` when RAM allows.

- **`ModuleNotFoundError` for dataset modules**  
  Verify the expected `movies_data` paths exist and were synced.

## Useful Scripts

- `run_eval_full_biases_windows.sh` — baseline sequential full evaluation
- `run_eval_full_biases_windows_parallel_seeded.sh` — seeded, isolated parallel evaluation
- `run_eval_w1_w3_parallel_isolated.sh` — focused isolated run for `W=1,3`

## License / Usage

This repository is prepared for academic reproducibility of the associated paper experiments. Add or update `LICENSE` before public release if required by your institution.