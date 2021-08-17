import numpy as np
import os
import pathlib

from petools.tools.estimate_tools import Human
from petools.model_tools.transformers.utils import HIP_ID
from ..utils import *


EPSILON = 1e-7


class PosePreprocessor:
    """
    This is a utility for normalizing data which feeds into classifier.

    """

    @staticmethod
    def init_from_lib():
        file_path = os.path.abspath(__file__)
        dir_path = pathlib.Path(file_path).parent
        data_stats_dir = os.path.join(dir_path, CLASSIFIER_STATS)
        mean_path = os.path.join(data_stats_dir, MEAN_2D)
        assert os.path.isfile(mean_path), f"Could not find {MEAN_2D} in {mean_path}."
        mean_2d = np.load(mean_path)

        std_path = os.path.join(data_stats_dir, STD_2D)
        assert os.path.isfile(std_path), f"Could not find {STD_2D} in {std_path}."
        std_2d = np.load(std_path)

        return PosePreprocessor(mean_2d, std_2d)

    def __init__(self, mean_2d: np.ndarray, std_2d: np.ndarray):
        self.mean2d = mean_2d.reshape(-1).astype('float32')
        self.std2d = std_2d.reshape(-1).astype('float32')

    def norm2d(self, human):
        human = self._take_array_from_human(human).reshape(-1)
        human -= self.mean2d
        human /= (self.std2d + EPSILON)
        return human

    def hip_shift(self, human):
        human = self._take_array_from_human(human)
        # Skip probability (last) values in last dimension
        human[:, :-1] -= human[HIP_ID, :-1]
        return human

    def _take_array_from_human(self, human) -> np.ndarray:
        if isinstance(human, Human):
            human = human.to_np(copy_if_cached=True)
        else:
            human = np.asarray(human)
        return human
