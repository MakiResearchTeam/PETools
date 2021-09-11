from dataclasses import dataclass
import numpy as np

from ..core import HumanRepresentation


class XYDecayingWeights:
    def __init__(self, shape, decay_steps, init=None):
        self._weights = init
        if self._weights is None:
            self._weights = np.ones(shape=shape, dtype='float32')
        self.decay_steps = decay_steps
        self.steps = decay_steps

    def start_decay(self):
        self.steps = 0

    def step(self):
        if self.steps < self.decay_steps:
            self.steps += 1

    @property
    def decay(self):
        return self.steps / self.decay_steps

    @property
    def weights(self):
        return self._weights * self.decay


@dataclass
class CustomRepresentation(HumanRepresentation):
    std_features: np.ndarray
    xy: np.ndarray
    std_xy: np.ndarray
    features_weights: np.ndarray
    xy_weights: XYDecayingWeights



