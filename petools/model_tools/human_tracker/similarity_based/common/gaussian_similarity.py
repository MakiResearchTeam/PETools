import numpy as np
from dataclasses import dataclass

from ..core import Similarity
from ..core import HumanRepresentation


@dataclass
class GaussianRepresentation(HumanRepresentation):
    std: np.ndarray


class Gauss(Similarity):
    def __call__(self, f1: GaussianRepresentation, f2: GaussianRepresentation, **kwargs) -> np.float:
        """
        Computes similarity, using gaussian kernel.

        Parameters
        ----------
        f1 : GaussianRepresentation
            This must be a representation from the registry. Contains kernel parameters.
        f2 : GaussianRepresentation
            This must be a newly computed human representation.

        Returns
        -------
        float
            Similarity value. The higher the value, the more similar the representations are.
        """
        kernel_mean = f1.features
        kernel_std = f1.std
        x = f2.features

        exp_arg = (kernel_mean - x) / kernel_std
        kernel_val = np.exp(-np.square(exp_arg))
        return np.mean(kernel_val)
