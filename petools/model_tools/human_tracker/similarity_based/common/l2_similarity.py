import numpy as np

from ..core import Similarity, HumanRepresentation


class L2(Similarity):
    def __call__(self, f1: HumanRepresentation, f2: HumanRepresentation, **kwargs) -> np.float:
        f1 = f1.features
        f2 = f2.features
        return np.mean(np.square(f1 - f2))