import numpy as np

from . import HumanProcessor
from .core import DataProcessor
from ...tools import Human


class Preprocess3D(DataProcessor):
    def __init__(self, human_processor: HumanProcessor):
        """
        A class for preprocessing 2d data before feeding the converter transformer.

        Parameters
        ----------
        human_processor : HumanProcessor
        """
        self.human_processor = human_processor

    def __call__(self, human: Human, skip_hip=False, **kwargs):
        source_resolution = kwargs['source_resolution']
        human_np = human.to_np()
        human, p = human_np[:, :2], human_np[:, 2:]
        p = np.concatenate([p]*2, axis=-1)
        human = self.shift_and_scale(human, source_resolution)
        human = self.human_processor.to_human36_format(human)
        p     = self.human_processor.to_human36_format(p)
        human = self.center_around_zero_point(human, p)
        if skip_hip:
            # Skip hip
            human = human[1:]
        human = human.reshape(-1)
        human = self.human_processor.norm2d(human)
        return human

    def shift_and_scale(self, human: np.ndarray, source_resolution):
        # Centers skeleton in a 700x700 square
        assert len(human.shape) == 2
        h, w = source_resolution
        human *= 700 / h
        # Shift the skeleton into the center of 1000x1000 square image
        """
        selected_x = human[:, 0][human[:, 0] != 0]
        left_x = 0
        if len(selected_x) != 0:
            left_x = np.min(selected_x)
        right_x = np.max(human[:, 0])
        width = right_x - left_x
        center = left_x + width / 2
        if np.max(human[:, 0]) > 900:
            shift = center - 500
            human[:, 0] -= shift
        elif np.min(human[:, 0]) < 100:
            shift = 500 - center
            human[:, 0] += shift
        """
        return human

    def center_around_zero_point(self, human, probabilities):
        mask = (probabilities > 1e-3).astype('float32')
        # Mask out absent points
        return (human - human[0:1]) * mask
