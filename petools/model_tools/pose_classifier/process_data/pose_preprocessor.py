import numpy as np
import os
import pathlib

from petools.tools.estimate_tools import Human
from ..utils import *


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
        if isinstance(human, Human):
            human = human.to_np()[:, :2]
        else:
            human = np.asarray(human)

        return (human.reshape(-1) - self.mean2d) / self.std2d

