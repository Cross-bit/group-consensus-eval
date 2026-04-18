"""
Rychlé testy hybridního mediátoru (úvodní sync podle počtu položek, vč. W < first_round_ration).
Bez scipy: před importem mediátoru se zaregistrují lehké stuby závislostí.
"""
from __future__ import annotations

import sys
import types
import unittest
from abc import ABC
from pathlib import Path
from unittest.mock import MagicMock

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_parent_chain(name: str) -> None:
    """Register parent packages so stub leaves can load; set __path__ for real dirs on disk."""
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent in sys.modules:
            continue
        m = types.ModuleType(parent)
        pkg_dir = _REPO_ROOT.joinpath(*parent.split("."))
        if pkg_dir.is_dir():
            m.__path__ = [str(pkg_dir)]
        sys.modules[parent] = m


def _install_lightweight_stubs() -> None:
    if "evaluation_frameworks.consensus_evaluation.consensus_mediator" in sys.modules:
        return

    _ensure_parent_chain("evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender_engine")
    rec = types.ModuleType(
        "evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender_engine"
    )

    class _B(ABC):
        pass

    rec.GeneralRecommendationEngineBase = _B
    rec.GroupRecommendationEngineBase = _B
    rec.RecommendationEngineGroupAllIndividualEaserUpdatable = type(
        "RecommendationEngineGroupAllIndividualEaserUpdatable", (_B,), {}
    )
    rec.RecommendationEngineGroupAllSameEaser = type("RecommendationEngineGroupAllSameEaser", (_B,), {})
    rec.RecommendationEngineGroupAllSameEaserWithFeedback = type(
        "RecommendationEngineGroupAllSameEaserWithFeedback", (_B,), {}
    )
    rec.RecommendationEngineIndividualEaser = type("RecommendationEngineIndividualEaser", (_B,), {})
    sys.modules[
        "evaluation_frameworks.consensus_evaluation.consensus_algorithm.recommender_engine"
    ] = rec

    _ensure_parent_chain(
        "evaluation_frameworks.consensus_evaluation.consensus_algorithm.redistribution_unit"
    )
    ru = types.ModuleType(
        "evaluation_frameworks.consensus_evaluation.consensus_algorithm.redistribution_unit"
    )

    class RedistributionContext:
        pass

    class RedistributionUnit:
        pass

    class SimplePriorityFunction:
        pass

    ru.RedistributionContext = RedistributionContext
    ru.RedistributionUnit = RedistributionUnit
    ru.SimplePriorityFunction = SimplePriorityFunction
    sys.modules[
        "evaluation_frameworks.consensus_evaluation.consensus_algorithm.redistribution_unit"
    ] = ru

    _ensure_parent_chain(
        "evaluation_frameworks.general_recommender_evaluation.algorithms.easer_cached"
    )
    ec = types.ModuleType(
        "evaluation_frameworks.general_recommender_evaluation.algorithms.easer_cached"
    )

    class EaserCached:
        pass

    ec.EaserCached = EaserCached
    sys.modules[
        "evaluation_frameworks.general_recommender_evaluation.algorithms.easer_cached"
    ] = ec

    _ensure_parent_chain("dataset.data_access")
    md = types.ModuleType("dataset.data_access")

    class MovieLensDatasetLoader:
        pass

    md.MovieLensDatasetLoader = MovieLensDatasetLoader
    sys.modules["dataset.data_access"] = md


_install_lightweight_stubs()

from evaluation_frameworks.consensus_evaluation.consensus_algorithm.models import Vote  # noqa: E402
from evaluation_frameworks.consensus_evaluation.consensus_mediator import (  # noqa: E402
    ConsensusMediatorHybridApproach,
    ThresholdPolicyStatic,
)


def _empty_votes(users: list[int]) -> dict[int, list[Vote]]:
    return {uid: [] for uid in users}


def _dummy_votes(users: list[int], tag: int) -> dict[int, list[Vote]]:
    return {uid: [Vote(10_000 * uid + tag, 1)] for uid in users}


class TestHybridPreamble(unittest.TestCase):
    def test_w1_fr3_three_preamble_rounds_then_async(self) -> None:
        users = [101, 102, 103]
        w, fr = 1, 3

        group_eng = MagicMock()
        group_eng.recommend_next_k.side_effect = [[10], [11], [12]]

        general = MagicMock()

        def _async_fill(uid: int, k: int, votes=None):
            return [2000 + uid * 100 + i for i in range(k)]

        general.recommend_next_k.side_effect = _async_fill

        red = MagicMock()
        red.get_user_redistribution_queue_size.return_value = 0
        red.get_redistributed_items.return_value = []

        m = ConsensusMediatorHybridApproach(
            users_ids=users,
            general_recommender=general,
            group_recommendation_engine=group_eng,
            redistribution_unit=red,
            first_round_ration=fr,
            threshold_policy=ThresholdPolicyStatic(1),
            window_size=w,
        )

        r1 = m.get_next_round_recommendation(_empty_votes(users))
        self.assertEqual(r1[users[0]], [10])
        self.assertEqual(group_eng.recommend_next_k.call_count, 1)

        r2 = m.get_next_round_recommendation(_dummy_votes(users, 1))
        self.assertEqual(r2[users[0]], [11])

        r3 = m.get_next_round_recommendation(_dummy_votes(users, 2))
        self.assertEqual(r3[users[0]], [12])
        self.assertEqual(group_eng.recommend_next_k.call_count, 3)

        r4 = m.get_next_round_recommendation(_dummy_votes(users, 3))
        self.assertEqual(len(r4[users[0]]), w)
        self.assertTrue(all(x >= 2000 for x in r4[users[0]]))
        general.recommend_next_k.assert_called()
        self.assertEqual(group_eng.recommend_next_k.call_count, 3)

    def test_w10_fr4_single_preamble_matches_old_shape(self) -> None:
        users = [1, 2, 3]
        w, fr = 10, 4

        group_eng = MagicMock()
        group_eng.recommend_next_k.return_value = [1, 2, 3, 4]

        general = MagicMock()
        general.recommend_next_k.side_effect = lambda uid, k, votes=None: [500 + uid + i for i in range(k)]

        red = MagicMock()
        red.get_user_redistribution_queue_size.return_value = 0
        red.get_redistributed_items.return_value = []

        m = ConsensusMediatorHybridApproach(
            users_ids=users,
            general_recommender=general,
            group_recommendation_engine=group_eng,
            redistribution_unit=red,
            first_round_ration=fr,
            threshold_policy=ThresholdPolicyStatic(2),
            window_size=w,
        )

        r1 = m.get_next_round_recommendation(_empty_votes(users))
        for uid in users:
            self.assertEqual(len(r1[uid]), w)
            self.assertEqual(r1[uid][:4], [1, 2, 3, 4])
            self.assertEqual(len(r1[uid][4:]), 6)

        group_eng.recommend_next_k.assert_called_once()
        self.assertEqual(group_eng.recommend_next_k.call_args[0][1], 4)

    def test_fr_below_one_raises(self) -> None:
        users = [1, 2, 3]
        with self.assertRaises(ValueError):
            ConsensusMediatorHybridApproach(
                users_ids=users,
                general_recommender=MagicMock(),
                group_recommendation_engine=MagicMock(),
                redistribution_unit=MagicMock(),
                first_round_ration=0,
                threshold_policy=ThresholdPolicyStatic(1),
                window_size=5,
            )


if __name__ == "__main__":
    unittest.main()
