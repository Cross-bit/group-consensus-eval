"""
Zpětná kompatibilita: dříve tento soubor procházel více ``first_round_ration`` —
to patří do ladění (``tune_hybrid_general_rec_individual.py``), ne do standardní eval.

Standardní eval s fixním poměrem prvních kol je
``eval_hybrid_general_rec_individual`` (hodnota ``first_r_ration`` v
``DEFAULT_SIGMOID_PARAMS`` pro dané ``w_size``).

Tento modul jen přesměruje ``autorun`` na stejnou třídu jako tam.
"""

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import autorun
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.eval_hybrid_general_rec_individual import (
    EvalHybridGeneralRecIndividual,
)

if __name__ == "__main__":
    autorun(EvalHybridGeneralRecIndividual)
