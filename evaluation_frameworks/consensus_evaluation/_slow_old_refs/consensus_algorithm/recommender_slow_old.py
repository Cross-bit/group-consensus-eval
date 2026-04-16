import time
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import numpy as np

from evaluation_frameworks.general_recommender_evaluation.algorithms.algorithm_base import RecAlgoIterator
from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.easer_group import GR_AggregatedProfilesUpdatable, GR_AggregatedRecommendations, TopKIterator
from evaluation_frameworks.consensus_evaluation.consensus_algorithm.models import Vote
from evaluation_frameworks.general_recommender_evaluation.algorithms.group_algorithms.interface import RecAlgoGroupAggregated, RecAlgoUpdatable
from movies_data.dataset.data_access import MovieLensDatasetLoader

class GeneralRecommendationEngineBase(ABC):

    @abstractmethod
    def get_all_recommended_items(self) -> Set[int]:
        "Returns set of all the items recommended to the client during"
        pass

    @abstractmethod
    def reset_iteration(self, users_ids: List[int], exclude_items: Optional[Set[int]] = None, agg_strategy: Optional[str] = 'mean') -> None:
        """
        Resets the recommender's internal state for a new recommendation session.

        Args:
            users_ids (List[int]): List of user IDs the recommender should consider.
            exclude_items (Optional[Set[int]]): Items to exclude from recommendations.
            agg_strategy (Optional[str]): Aggregation strategy to use (e.g. 'mean', 'min', etc.).

        Note:
            This method should be called whenever the user group or configuration changes,
            to ensure the recommender reflects the new context.
        """
        pass

    @abstractmethod
    def recommend_next_k(self, user_id: int, size: int, lastVotes: List[Vote] = None) -> List[int]:
        pass

class GroupRecommendationEngineBase(ABC):

    @abstractmethod
    def get_all_recommended_items(self) -> Set[int]:
        "Returns set of all the items recommended to the client during"
        pass

    @abstractmethod
    def reset_iteration(self, users_ids: List[int], exclude_items: Optional[Set[int]] = None, agg_strategy: Optional[str] = 'mean') -> None:
        """
        Resets the recommender's internal state for a new recommendation session.

        Args:
            users_ids (List[int]): List of user IDs the recommender should consider.
            exclude_items (Optional[Set[int]]): Items to exclude from recommendations.
            agg_strategy (Optional[str]): Aggregation strategy to use (e.g. 'mean', 'min', etc.).

        Note:
            This method should be called whenever the user group or configuration changes,
            to ensure the recommender reflects the new context.
        """
        pass

    @abstractmethod
    def recommend_next_k(self, user_id: List[int], size: int, lastVotes: Dict[int, List[Vote]] = None) -> List[int]:
        pass


# ==========================================================
# Recommendation engines for async consensus algorithm
# ==========================================================



#
# Generates single-user recommendation to each member (the members data are not aggregated)
#
class RecommendationEngineIndividualEaser(GeneralRecommendationEngineBase):

    def __init__(self, users_ids: List[int], model_iterator: RecAlgoIterator):
        """ Each user gets his own single-user recommender recommendation.

        Args:
            user_ids (List[str]): Group user IDs.
            easer (EaserCached): Easer algorithm implementation.
        """

        self.model = model_iterator
        self.easer_iterators = None
        self.reset_iteration(users_ids)

        super().__init__()

    def get_all_recommended_items(self) -> Set[int]:
        return self.all_recommended_item_ids

    def reset_iteration(self, users_ids: List[int], exclude_items: Optional[Set[int]] = None, agg_strategy: Optional[str] = 'mean') -> None:
        """
            Resets the recommender's internal state for a new recommendation session.

            Args:
                users_ids (List[int]): List of user IDs the recommender should consider.
                exclude_items (Optional[Set[int]]): Items to exclude from recommendations.
                agg_strategy (Optional[str]): Aggregation strategy to use (e.g. 'mean', 'min', etc.).

            Note:
                This method should be called whenever the user group or configuration changes,
                to ensure the recommender reflects the new context.
        """
        self.easer_iterators = { user_id: self.model.top_k_iterator(user_id, exclude_items) for user_id in users_ids }
        self.all_recommended_item_ids = exclude_items

    def recommend_next_k(self, user_id: int, size: int, lastVotes: List[Vote] = None) -> List[int]:
        user_recommendation = [next(self.easer_iterators[user_id]) for _ in range(size)]
        return user_recommendation

#
# Generates different GROUP recommendation to each member (the members data are aggregated)
#
class RecommendationEngineGroupAllIndividualEaser(GeneralRecommendationEngineBase):

    def __init__(self, users_ids: List[int], model: RecAlgoGroupAggregated):
        self.all_recommended_item_ids = []
        self.model: RecAlgoGroupAggregated = model
        self.current_iterator = None
        self.users_ids = users_ids
        # One shared slate per mediator round (see begin_new_recommendation_round).
        self._group_sync_slate: Optional[List[int]] = None

        self.reset_iteration(users_ids)

        super().__init__()

    def get_all_recommended_items(self) -> Set[int]:
        return self.all_recommended_item_ids

    def begin_new_recommendation_round(self) -> None:
        """Invalidate cached slate so the next recommend_next_k fills a fresh group list for this round."""
        self._group_sync_slate = None

    def reset_iteration(self, users_ids: List[int], exclude_items: Optional[Set[int]] = None, agg_strategy: Optional[str] = 'mean') -> None:
        """
        Resets the recommender's internal state for a new recommendation session.

        Args:
            users_ids (List[int]): List of user IDs the recommender should consider.
            exclude_items (Optional[Set[int]]): Items to exclude from recommendations.
            agg_strategy (Optional[str]): Aggregation strategy to use (e.g. 'mean', 'min', etc.).

        Note:
            This method should be called whenever the user group or configuration changes,
            to ensure the recommender reflects the new context.
        """

        if users_ids == []:
            return

        self.current_iterator = self.model.top_k_iterator(users_ids, agg_strategy, exclude_items)
        self.all_recommended_item_ids = exclude_items
        self.users_ids = users_ids
        self._group_sync_slate = None

    def recommend_next_k(self, user_id: int, size: int, last_votes: List[Vote] = None) -> List[int]:
        # Shared group slate: mediator calls recommend_next_k once per member per round.
        # Without caching, each call would advance the group iterator and members would see
        # disjoint items in the same round (consensus then becomes extremely unlikely for small W).
        if self._group_sync_slate is None or len(self._group_sync_slate) != size:
            self._group_sync_slate = [next(self.current_iterator) for _ in range(size)]
        self.last_recommendation_buffer = list(self._group_sync_slate)
        return self.last_recommendation_buffer


# For async hybrid with group recommender and feedback
class RecommendationEngineGroupAllIndividualEaserUpdatable(GeneralRecommendationEngineBase):

    def __init__(self, users_ids: List[int], updatable_group_model: GR_AggregatedProfilesUpdatable):
        self.all_recommended_item_ids = []
        self.updatable_group_model: GR_AggregatedProfilesUpdatable = updatable_group_model
        self.current_iterator = None
        self.users_ids = users_ids

        self.served_users_in_current_round = 0
        self.reset_iteration(users_ids)

        super().__init__()

    def get_all_recommended_items(self) -> Set[int]:
        return self.all_recommended_item_ids

    def reset_iteration(self, users_ids: List[int], exclude_items: Optional[Set[int]] = None, agg_strategy: Optional[str] = 'mean') -> None:
        """
        Resets the recommender's internal state for a new recommendation session.

        Args:
            users_ids (List[int]): List of user IDs the recommender should consider.
            exclude_items (Optional[Set[int]]): Items to exclude from recommendations.
            agg_strategy (Optional[str]): Aggregation strategy to use (e.g. 'mean', 'min', etc.).

        Note:
            This method should be called whenever the user group or configuration changes,
            to ensure the recommender reflects the new context.
        """

        if users_ids == []:
            return

        if exclude_items:
            self.all_recommended_item_ids = list(exclude_items)

        self.users_ids = users_ids

    def update_model(self, last_votes: Dict[int, List[Vote]]):
        if len(last_votes) > 0:
            mapped = self.normalize_feedback_to_stars(last_votes)
            self.updatable_group_model.update_group_with_votes(mapped, reduce="mean")

        # make sure we are really tracking all the items
        all_rec_items_ids_set = set(self.all_recommended_item_ids)
        for votes in last_votes.values():
            new_ids = {v.id for v in votes if v.id not in all_rec_items_ids_set}
            self.all_recommended_item_ids.extend(new_ids)

    def recommend_next_k(self, user_id: int, size: int, last_votes: List[Vote] = None) -> List[int]:

        # UPDATING: we leave the update on the caller using -- update_model -- method

        new_rec = self.updatable_group_model.recommend_group_top_k(self.users_ids, size, exclude=set(self.all_recommended_item_ids))
        self.all_recommended_item_ids.extend(new_rec)

        self.served_users_in_current_round += 1
        return new_rec

    def normalize_feedback_to_stars(
        self,
        user_votes: Dict[int, List["Vote"]],
        *,
        star_min: float = 1.0,
        star_max: float = 5.0,
    ) -> Dict[int, Dict[int, float]]:
        """
        Convert per-user lists of Vote objects into a nested dict with star ratings.

        Args:
            user_votes: { user_id: [Vote(item_id, value in {-1,0,1}), ...], ... }
            star_min:   value for -1 feedback (default 1.0)
            star_max:   value for +1 feedback (default 5.0)

        Returns:
            { user_id: { item_id: mapped_star (float) }, ... }
            Mapping uses midpoint for zero.
        """
        midpoint = 0.5 * (star_min + star_max)
        out: Dict[int, Dict[int, float]] = {}
        # print(user_votes)
        for uid, votes in user_votes.items():
            mapped: Dict[int, float] = {}
            for v in votes:
                val = float(getattr(v, "value"))
                iid = int(getattr(v, "id"))
                if val > 0:
                    mapped[iid] = float(star_max)
                elif val < 0:
                    mapped[iid] = float(star_min)
                else:  # val == 0
                    mapped[iid] = float(midpoint)
            if mapped:  # skip empty
                out[uid] = mapped

        return out



### =============================================================================================================


# ==========================================================
# Recommendation engines for sync consensus algorithm
# ==========================================================

#
# Generates same recommendation to each member (the members data are aggregated)
#
class RecommendationEngineGroupAllSameEaser(GroupRecommendationEngineBase):

    def __init__(self, users_ids: List[int], ratings_matrix: pd.DataFrame, model: RecAlgoGroupAggregated, default_strategy='mean'):
        self.ratings_matrix:pd.DataFrame = ratings_matrix
        self.all_recommended_item_ids = []
        self.model: RecAlgoGroupAggregated = model
        self.current_iterator = None
        self.users_ids = users_ids
        self.last_recommendation_buffer = [] # stores last group recommendation
        self.number_of_users_recommended_to_in_round = 0
        self.users_served_in_this_round = set(users_ids) # all the users we served in current round

        self.reset_iteration(users_ids, agg_strategy=default_strategy)

        super().__init__()

    def get_all_recommended_items(self) -> Set[int]:
        return self.all_recommended_item_ids

    def reset_iteration(self, users_ids: List[int], exclude_items: Optional[Set[int]] = None, agg_strategy: Optional[str] = 'mean') -> None:
        """
        Resets the recommender's internal state for a new recommendation session.

        Args:
            users_ids (List[int]): List of user IDs the recommender should consider.
            exclude_items (Optional[Set[int]]): Items to exclude from recommendations.
            agg_strategy (Optional[str]): Aggregation strategy to use (e.g. 'mean', 'min', etc.).

        Note:
            This method should be called whenever the user group or configuration changes,
            to ensure the recommender reflects the new context.
        """

        self.current_iterator = self.model.top_k_iterator(users_ids, agg_strategy, exclude_items)
        self.all_recommended_item_ids = exclude_items
        self.users_ids = users_ids
        self.last_recommendation_buffer = [] # stores last group recommendation
        self.users_served_in_this_round = set(users_ids)

    def recommend_next_k(self, user_id: List[int], size: int, lastVotes: Dict[int, List[Vote]] = None) -> List[int]:
        self.last_recommendation_buffer = [next(self.current_iterator) for _ in range(size)]
        return self.last_recommendation_buffer

    def _all_group_members_served(self) -> bool:
        return len(self.users_served_in_this_round) == len(self.users_ids)

    def _update_served_users(self, user_id: int):
        self.users_served_in_this_round.add(user_id)


class RecommendationEngineGroupAllSameEaserWithFeedback(GroupRecommendationEngineBase):

    def __init__(self, users_ids: List[int], model: GR_AggregatedProfilesUpdatable, default_strategy='mean'):
        self.all_recommended_item_ids = []
        self.model: GR_AggregatedProfilesUpdatable = model
        self.current_iterator = None
        self.users_ids = users_ids
        self.last_recommendation_buffer = [] # stores last group recommendation
        self.number_of_users_recommended_to_in_round = 0
        self.users_served_in_this_round = set(users_ids) # all the users we served in current round

        self.reset_iteration(users_ids, agg_strategy=default_strategy)

        super().__init__()

    def get_all_recommended_items(self) -> Set[int]:
        return self.all_recommended_item_ids

    def reset_iteration(self, users_ids: List[int], exclude_items: Optional[Set[int]] = None, agg_strategy: Optional[str] = 'mean') -> None:
        pass

    def recommend_next_k(self, user_ids: List[int], size: int, lastVotes: Dict[int, List[Vote]] = None) -> List[int]:

        if len(self.all_recommended_item_ids) > 0:
            mapped = self.normalize_feedback_to_stars(lastVotes) # {-1,0,1} -> 1..5 (0 -> midpoint)
            self.model.update_group_with_votes(mapped, reduce="mean")

        new_rec = self.model.recommend_group_top_k(user_ids, size, exclude=set(self.all_recommended_item_ids))
        self.all_recommended_item_ids.extend(new_rec)

        return new_rec

    def normalize_feedback_to_stars(
        self,
        user_votes: Dict[int, List["Vote"]],
        *,
        star_min: float = 1.0,
        star_max: float = 5.0,
    ) -> Dict[int, Dict[int, float]]:
        """
        Convert per-user lists of Vote objects into a nested dict with star ratings.

        Args:
            user_votes: { user_id: [Vote(item_id, value in {-1,0,1}), ...], ... }
            star_min:   value for -1 feedback (default 1.0)
            star_max:   value for +1 feedback (default 5.0)

        Returns:
            { user_id: { item_id: mapped_star (float) }, ... }
            Mapping uses midpoint for zero.
        """
        midpoint = 0.5 * (star_min + star_max)
        out: Dict[int, Dict[int, float]] = {}
        # print(user_votes)
        for uid, votes in user_votes.items():
            mapped: Dict[int, float] = {}
            for v in votes:
                val = float(getattr(v, "value"))
                iid = int(getattr(v, "id"))
                if val > 0:
                    mapped[iid] = float(star_max)
                elif val < 0:
                    mapped[iid] = float(star_min)
                else:
                    mapped[iid] = float(midpoint)
            if mapped:  # skip empty
                out[uid] = mapped

        return out


#    def normalize_feedback_to_stars(
#        self,
#        user_item_interactions: Dict[int, Dict[int, float]],
#        *,
#        star_min: float = 1.0,
#        star_max: float = 5.0,
#        ) -> Dict[int, Dict[int, float]]:
#            """
#            Maps {-1,0,1} -> [star_min, star_max].
#            -1 -> star_min
#            +1 -> star_max
#            0 -> midpoint
#            """
#
#            midpoint = 0.5 * (star_min + star_max)
#            out: Dict[int, Dict[int, float]] = {}
#
#            for uid, items in user_item_interactions.items():
#                mapped: Dict[int, float] = {}
#                for item_id, v in items.items():
#                    if v > 0:
#                        mapped[item_id] = float(star_max)
#                    elif v < 0:
#                        mapped[item_id] = float(star_min)
#                    else:
#                        mapped[item_id] = float(midpoint)
#
#                if mapped:  # skip empty
#                    out[uid] = mapped
#            return out
#
#



if __name__ == "__main__":

    d_loader = MovieLensDatasetLoader()
    _, ratings_matrix = d_loader.load_data(True)
    ratings_matrix_np = ratings_matrix.to_numpy()

    user_ids = [42, 24, 5, 6]

### Individual recommendation test

#    easer_cached = EaserCached()
#    easer_cached.fit(ratings_matrix)
#
#    easer_cached.precalculate_scores(user_ids)
#    groupRecEaser = RecommendationEngineIndividualEaser(ratings_matrix, easer_cached)
#
#    print(groupRecEaser.recommend_next_k(user_ids, 5))
#    print(groupRecEaser.recommend_next_k(user_ids, 5))
#    print(groupRecEaser.recommend_next_k(user_ids, 5))

### Same group recommendation for all test

    groupRecEaser = GR_AggregatedRecommendations()
    groupRecEaser.fit(ratings_matrix)

    user_ids = [42, 24, 5, 6]
    recEngine = RecommendationEngineGroupAllSameEaser(user_ids, ratings_matrix, groupRecEaser)

    print(recEngine.recommend_next_k(42, 5))
    print(recEngine.recommend_next_k(24, 5))
    print(recEngine.recommend_next_k(42, 5))
    print(recEngine.recommend_next_k(5, 5))
    print(recEngine.recommend_next_k(6, 5))

    recEngine.reset_iteration([42, 24], agg_strategy = "median")

    #print(recEngine.recommend_next_k(user_ids, 5))
    #print(recEngine.recommend_next_k(user_ids, 5))