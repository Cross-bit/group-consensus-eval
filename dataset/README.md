# Dataset layout and usage

This project expects MovieLens data under `dataset/` in the following structure:

```text
dataset/
  data_access.py
  ml-1m/
    movies.csv
    ratings.csv
    ... (optional local extras)
  ml-32m/
    movies.csv
    ratings.csv
    README.txt
    checksums.txt
    ... (optional: tags.csv, links.csv)
```

## Which files are required

- `movies.csv` and `ratings.csv` are required by `MovieLensDatasetLoader`.
- `tags.csv` and `links.csv` are optional for this repo (kept for reference).
- For large datasets, `ml-32m/sparse-cached-ml-32m.pkl` can be created automatically when using sparse cache mode.

## `data_access.py` quick start

Import:

```python
from dataset.data_access import MovieLensDatasetLoader
```

### 1) Dense matrix load (small data, e.g. `ml-1m`)

```python
loader = MovieLensDatasetLoader(dataset_dir="ml-1m", movies_file="movies.csv", ratings_file="ratings.csv")
movies_df, ratings_df = loader.load_data(fill_zeroes=False)
```

- `movies_df`: movie metadata (`movieId`, `title`, `genres`)
- `ratings_df`: pivot matrix (`userId` x `movieId`)
- Warning: dense matrix is memory-heavy; do not use for `ml-32m`.

### 2) Sparse load (recommended for `ml-32m`)

```python
loader = MovieLensDatasetLoader(dataset_dir="ml-32m", movies_file="movies.csv", ratings_file="ratings.csv")
movies_df, ratings_csr, user_id_map, movie_id_map = loader.load_sparse_ratings(use_cache=True)
```

- `ratings_csr`: `scipy.sparse.csr_matrix` (`users x movies`)
- `user_id_map`: sparse row index -> real `userId`
- `movie_id_map`: sparse col index -> real `movieId`
- `use_cache=True`: reads/writes `sparse-cached-<dataset>.pkl` in the same dataset folder

### 3) Optional filtering/splitting helpers

- `filter_csr_by_interaction_thresholds(...)`
- `split_csr_train_val_test(...)`
- `split_csr_by_users_full_mapping(...)`
- `get_surprise_trainset()` (after `load_data()`)

## Notes

- Paths in `MovieLensDatasetLoader` are resolved relative to the `dataset/` directory.
- The old script `load_dataset_OBSOLETE.py` is not used anymore and has been removed.
