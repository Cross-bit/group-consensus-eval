#!/bin/python3
import os
import sys
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

import surprise
from surprise.model_selection import KFold

# ===================================
# DESCRIPTION
# ===================================
# Simple evaluation framework based on surprise
# to test different algorithms.
#

from surprise import Dataset, Reader
from scipy.sparse import dok_matrix
from abc import ABC, abstractmethod

class SurpriseRatingBasedEvaluation(ABC):
    """Defines base class for the recommendation algorithms evaluations. """

    def __init__(self, rating_matrix: pd.DataFrame, test_size: float = 0.2, rating_scale=(1, 5)):
        self.rating_matrix = rating_matrix

        self.test_size = test_size
        self.rating_scale = rating_scale

        self.trainset = None
        self.testset = None
        self.train_matrix = None
        self.algo = None

        self._initiate_train_matrix()

    @abstractmethod
    def fit(self):
        """ Train the model.
            Example:
        """
        pass

    @abstractmethod
    def evaluate(self):
        """Run evaluation and return results."""
        pass

    def evaluate_crossval(self, n_splits=5):
        ratings_ndarray = self.rating_matrix.to_numpy()
        rows, cols = ratings_ndarray.nonzero()

        ratings = pd.DataFrame({
            "user": rows,
            "item": cols,
            "rating": ratings_ndarray[rows, cols]
        })

        reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(ratings[['user', 'item', 'rating']], reader)

        kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        results = []

        for trainset, testset in kf.split(data):
            self.trainset = trainset
            self.testset =  testset
            self._initiate_train_matrix_from_existing_trainset()
            self.fit()
            results.append(self.evaluate())

        return self._average_results(results)


    def _initiate_train_matrix(self):
        """Initiates matrix of train data."""
        self._initiate_test_sets()
        self._initiate_train_matrix_from_existing_trainset()

    def _initiate_train_matrix_from_existing_trainset(self):
        """Initiates matrix of train data using already created test sets(used for KFold evaluation)."""
        num_users = self.trainset.n_users
        num_items = self.trainset.n_items
        self.train_matrix = dok_matrix((num_users, num_items))

        for uid, iid, rating in self.trainset.all_ratings():
            self.train_matrix[int(uid), int(iid)] = rating


    def _average_results(self, results):
        """Averages results from the cross validation."""
        avg = {}
        for key in results[0]:
            avg[key] = np.mean([r[key] for r in results])
        return avg

    def _initiate_test_sets(self) -> tuple:
        """Convert original rating matrix into the numpy nd array.

        Args:
            test_size (float, optional): Percentage of data used for testing. Defaults to 0.2.¨
        """

        ratings_ndarray = self.rating_matrix.to_numpy()
        rows, cols = ratings_ndarray.nonzero()

        ratings = pd.DataFrame({
            "user": rows,
            "item": cols,
            "rating": ratings_ndarray[rows, cols]
        })
        reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(ratings[['user', 'item', 'rating']], reader)

        self.trainset, self.testset = surprise.model_selection.train_test_split(data, test_size=self.test_size)

        return (self.trainset, self.testset)
