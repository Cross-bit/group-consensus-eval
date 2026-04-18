#!/bin/python3
"""
Compatibility shim.

Historically, `SurpriseRatingBasedEvaluation` žila přímo v tomto modulu.
Teď je skutečná implementace v:
`evaluation_frameworks.general_recommender_evaluation.evaluation.surprise_rating_eval`.

Tenhle soubor jen re‑exportuje třídu, aby staré importy fungovaly:
`from evaluation_frameworks.general_recommender_evaluation.evaluation import SurpriseRatingBasedEvaluation`
"""

from evaluation_frameworks.general_recommender_evaluation.evaluation.surprise_rating_eval import (
    SurpriseRatingBasedEvaluation,
)

__all__ = ["SurpriseRatingBasedEvaluation"]
