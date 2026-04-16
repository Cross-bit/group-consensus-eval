import random
from typing import List, Tuple
# DESCRIPTION



class GroupsEvaluationSetsSplitter:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def split_by_counts(
        self,
        groups: List[List[int]],
        val_count: int,
        test_count: int
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """
            Randomly splits a list of user groups into train/validation/test sets based on fixed counts.

            The number of validation and test groups is specified explicitly.
            The remaining groups are assigned to the training set.

            :param groups: Input list of groups (each group is a list of user IDs).
            :param val_count: Number of groups to assign to the validation set.
            :param test_count: Number of groups to assign to the test set.
            :return: A tuple of (train_groups, val_groups, test_groups)
        """

        n = len(groups)
        if n == 0:
            return [], [], []

        # When there are fewer groups than val_count + test_count, filling validation first
        # would leave test empty (eval defaults to test split → 0 groups). Split proportionally.
        total_req = val_count + test_count
        if total_req > 0 and n < total_req:
            val_use = (n * val_count) // total_req
            test_use = n - val_use
        else:
            val_use = min(val_count, n)
            test_use = min(test_count, n - val_use)

        random.seed(self.seed)
        groups_shuffled = groups.copy()
        random.shuffle(groups_shuffled)

        val_groups = groups_shuffled[:val_use]
        test_groups = groups_shuffled[val_use : val_use + test_use]
        train_groups = groups_shuffled[val_use + test_use :]

        return train_groups, val_groups, test_groups
