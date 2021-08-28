import numpy as np
import os
import pathlib

from petools.tools.estimate_tools import Human
from ..utils import *


EPSILON = 1e-7
PATH_DATA = None


class PosePreprocessor:
    """
    This is a utility for normalizing data which feeds into classifier.

    """

    # TODO: Delete this method. Method only used as debug tool
    @staticmethod
    def set_data_path(path: str):
        global PATH_DATA
        PATH_DATA = path

    @staticmethod
    def init_from_lib():
        if PATH_DATA is not None:
            mean_path = os.path.join(PATH_DATA, MEAN_2D)
            assert os.path.isfile(mean_path), f"Could not find {MEAN_2D} in {mean_path}."
            mean_2d = np.load(mean_path)

            std_path = os.path.join(PATH_DATA, STD_2D)
            assert os.path.isfile(std_path), f"Could not find {STD_2D} in {std_path}."
            std_2d = np.load(std_path)
        else:
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
        print('inside preprocess: ', human.shape)
        print('loaded mean: ', self.mean2d.shape, ' loaded std: ', self.std2d.shape)
        human -= self.mean2d
        human /= (self.std2d + EPSILON)
        return human

