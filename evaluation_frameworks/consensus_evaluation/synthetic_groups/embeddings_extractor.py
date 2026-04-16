from typing import Dict

import numpy as np


class EmbeddingExtractor:
    def __init__(self, model_type: str = "lightfm"):
        self.model_type = model_type

    def extract_user_embeddings(self, model, user_id_map: Dict[int, int] = None) -> Dict[int, np.ndarray]:
        if self.model_type == "lightfm":
            return self._extract_lightfm_user_embeddings(model, user_id_map)
        else:
            raise NotImplementedError(f"Model type {self.model_type} not supported.")

    def _extract_lightfm_user_embeddings(self, model, user_id_map):
        n_users = model.user_embeddings.shape[0]
        user_embeddings = {}

        for user_idx in range(n_users):
            user_id = user_id_map[user_idx]
            user_embeddings[user_id] = model.user_embeddings[user_idx]

        return user_embeddings