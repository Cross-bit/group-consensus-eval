#!/bin/python3
import pandas as pd
from typing import Tuple
import os

# ====================================
# DESCRIPTION -- OBSOLETE!!! (User data_access.py instead)
# ====================================
# MovieLens dataset loader.
#

DATASET_DIR = os.path.join(os.path.dirname(__file__), "ml-1m")

def load_ml1() -> Tuple[pd.DataFrame,  pd.DataFrame]:

    movies_df = pd.read_csv(f'{DATASET_DIR}/movies.dat',
                        sep='::',
                        names=['movieId', 'title', 'genres'],
                        engine='python',
                        encoding='latin1')

    ratings_df = pd.read_csv(f'{DATASET_DIR}/ratings.dat',
                        sep='::',
                        names=['userId', 'movieId', 'ratings', 'timestamp'],
                        engine='python')

    # Drop timestamp column
    ratings_df.drop('timestamp', axis=1, inplace=True)

    # Create full ratings matrix
    ratings_df = ratings_df.pivot(index='userId', columns='movieId', values='ratings')

    return (movies_df, ratings_df)


def filter_by_popularity(dataset: Tuple[pd.DataFrame,pd.DataFrame], pop_threshold: float) -> Tuple[pd.DataFrame,  pd.DataFrame]:

    movies_df, ratings_df = dataset
    movie_popularity = ratings_df.notna().sum(axis=0)  # Use sum() instead of count() for clarity

    valid_movies_count =  int(movie_popularity.size * (pop_threshold / 100)) # Select top pop_threshold % of the movies

    filtered_movies = movie_popularity.sort_values(ascending=False).iloc[:valid_movies_count]

    movies_df_new = movies_df.set_index("movieId").loc[movies_df.set_index("movieId").index.intersection(filtered_movies.index), :]
    ratings_df_new = ratings_df[ratings_df.index.intersection(filtered_movies.index)]

    return (movies_df_new, ratings_df_new)

def user_genre_avg_ratings(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Convert ratings_df to long form
    ratings_long = ratings_df.reset_index().melt(id_vars='userId', var_name='movieId', value_name='rating')
    ratings_long.dropna(subset=['rating'], inplace=True)

    # Step 2: Merge with movie genres
    movies_genres = movies_df[['genres']].reset_index()  # movieId as column now
    merged = ratings_long.merge(movies_genres, on='movieId', how='left')

    # Step 3: One-hot encode genres
    genres_dummies = merged['genres'].str.get_dummies(sep='|')
    merged = pd.concat([merged, genres_dummies], axis=1)

    # Step 4: Multiply genre dummies by rating
    for genre in genres_dummies.columns:
        merged[genre] = merged[genre] * merged['rating']

    # Step 5: Compute sums and counts
    genre_sum = merged.groupby('userId')[genres_dummies.columns].sum()

    # fix: cast to bool before groupby
    genre_presence = merged[genres_dummies.columns].astype(bool)
    genre_count = genre_presence.groupby(merged['userId']).sum()

    # Step 6: Divide to get averages
    genre_avg = genre_sum / genre_count.replace(0, pd.NA)

    return genre_avg