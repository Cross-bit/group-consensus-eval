# Movie Data Notes

This repository does **not** version large MovieLens 32M data artifacts.

## Required Local Files (not in Git)

Place these files under `movies_data/dataset/ml-32m/` locally:

- `movies.csv`
- `ratings.csv`
- `links.csv`
- `tags.csv`
- `genome-scores.csv` (if used by your pipeline)
- `genome-tags.csv` (if used by your pipeline)

Optional local cache files that may be generated during evaluation:

- `sparse-cached-ml-32m.pkl`
- other `*.pkl` files in `ml-32m/`

These are intentionally ignored by `.gitignore` to keep the repository lightweight and publishable.

## Dataset Source

Download MovieLens data from the official GroupLens website:

- <https://grouplens.org/datasets/movielens/>

Use the dataset version expected by the evaluation scripts (32M in this project).
