from typing import List, Optional, Set, Tuple
from evaluation_frameworks.general_recommender_evaluation.iterators.iterator_base import BaseRecommendationIterator

#class StaticTopKIterator(BaseRecommendationIterator):
#    def __init__(
#        self,
#        item_scores: List[Tuple[str, float]],
#        exclude: Optional[Set[int]] = None,
#    ):
#        """
#            Args:
#                item_scores (List[Tuple[item_id, score]]): List of items with their predicted score.
#                exclude (Optional[Set[str]]): Set of item Ids to skip during iteration (e.g. those we already recommended etc...).
#        """
#
#        self.exclude = exclude or set()
#        self.sorted_items = sorted(item_scores, key=lambda x: x[1], reverse=True)
#        self.position = 0
#
#    def __iter__(self):
#        return self
#
#    def __next__(self) -> str:
#        while self.position < len(self.sorted_items):
#            item_id, _ = self.sorted_items[self.position]
#            self.position += 1
#            if item_id not in self.exclude:
#                return item_id
#
#        raise StopIteration