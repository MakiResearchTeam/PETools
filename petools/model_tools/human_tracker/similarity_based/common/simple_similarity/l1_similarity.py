import numpy as np

from petools.model_tools.human_tracker.similarity_based.core import HumanRepresentation
from .simple_similarity import SimilarityMeasure


class L1(SimilarityMeasure):
    def __call__(self, f1: HumanRepresentation, f2: HumanRepresentation, **kwargs) -> np.float:
        f1 = f1.features
        f2 = f2.features
        return np.mean(np.abs(f1 - f2))
