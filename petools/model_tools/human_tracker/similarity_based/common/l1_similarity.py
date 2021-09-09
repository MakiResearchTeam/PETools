import numpy as np

from ..core import Similarity


class L1(Similarity):
    def __call__(self, f1, f2, **kwargs) -> object:
        return np.mean(np.abs(f1 - f2))
