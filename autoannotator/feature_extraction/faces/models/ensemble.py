from typing import Any, Dict, List

import numpy as np
from autoannotator.feature_extraction.core.feature_extractor import BaseFeatureExtrator


class FaceFeatureExtractionEnsemle:
    def __init__(self, models: List[BaseFeatureExtrator], reduce: str="concat") -> None:
        assert models is not None and len(models) > 0
        self.models = models
        self.__check_model_names_unique()
        self.reduce_type = reduce
        
    def __check_model_names_unique(self):
        known_names = set()
        for model in self.models:
            name = model.config.name
            if name in known_names:
                raise Exception(f"Model with name {name} already exists in the face descriptor extraction ensemble")
            known_names.add(name)
        
    def __call__(self, image: np.ndarray) -> Any:
        embeddings = {}
        for model in self.models:
            embeddings[model.config.name] = model(image)
        return self.reduce(embeddings)
        
    def reduce(self, embeddings: Dict[str, np.ndarray]):
        match self.reduce_type:
            case "concat":
                values = list(embeddings.values())
                if not values:
                    raise ValueError("No embeddings provided for reduction")

                first = values[0]
                if first.ndim == 1:
                    result_embedding = np.concatenate(values, axis=0)
                elif first.ndim == 2:
                    batch_size = first.shape[0]
                    if any(val.ndim != 2 or val.shape[0] != batch_size for val in values):
                        raise ValueError("All embeddings must share the same batch dimension")
                    result_embedding = np.concatenate(values, axis=1)
                else:
                    raise ValueError(
                        "Unsupported embedding dimensions for concatenation"
                    )
                return result_embedding
            case _:
                raise ValueError(f"Unknown reduce type {self.reduce_type}")