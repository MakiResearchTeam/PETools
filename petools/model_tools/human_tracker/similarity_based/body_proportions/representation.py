from dataclasses import dataclass
import numpy as np

from ..core import HumanRepresentation


class XYDecayingWeights:
    def __init__(self, shape, decay_steps, init=None):
        """
        Used for switching priority between different tracking criteria (mean point and body proportions).

        Parameters
        ----------
        shape : list or int
            Used for default initialization with ones.
        decay_steps : int
            Number of frames the weights will decay.
        init : np.ndarray
            Custom weights values.
        """
        self._weights = init
        if self._weights is None:
            self._weights = np.ones(shape=shape, dtype='float32')
        self.decay_steps = decay_steps
        self.steps = decay_steps

    def start_decay(self):
        """
        Turns on weights decay.
        """
        self.steps = 0

    def step(self):
        """
        Updates weights decay.
        """
        if self.steps < self.decay_steps:
            self.steps += 1

    @property
    def decay(self):
        return (self.steps / self.decay_steps) ** 2

    @property
    def weights(self):
        return self._weights * self.decay


@dataclass
class FancyRepresentation(HumanRepresentation):
    std_features: np.ndarray
    xy: np.ndarray
    std_xy: np.ndarray
    features_weights: np.ndarray
    xy_weights: XYDecayingWeights
