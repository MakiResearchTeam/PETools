import numpy as np

from ..core import Similarity


class L2(Similarity):
    def __call__(self, f1, f2, **kwargs) -> object:
        return np.linalg.norm(f1 - f2)
