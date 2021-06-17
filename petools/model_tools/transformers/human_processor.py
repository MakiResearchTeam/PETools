import numpy as np
import os
import pathlib

from petools.tools.estimate_tools import Human
from .utils import *


class HumanProcessor:
    """
    This is a utility for normalizing/denormalizing data from Human3.6M dataset.
    """
    SELECT2D = init_selector()
    INDICES3DTO32POINTS = INDICES3DTO32POINTS
    PRODUCTION_TO_HUMAN36 = PRODUCTION_TO_HUMAN36
    H32_TO_HPROD_3D = H32_TO_HPROD_3D

    @staticmethod
    def to_human36_format(production_points):
        """
        Converts 23 production points to 16 human points. Can be used only for 2d points.
        """
        human36_points = np.zeros((H3P6_2D_NUM, 2), dtype='float32')
        for i, j in HumanProcessor.PRODUCTION_TO_HUMAN36:
            human36_points[j] = production_points[i]
        return human36_points[HumanProcessor.SELECT2D]

    @staticmethod
    def to_human36_format3d(production_points):
        """
        Converts 23 production points to 16 human 3d points. Can be used only for 3d points.
        """
        human36_points = np.zeros((H3P6_2D_NUM, 3), dtype='float32')
        for i, j in HumanProcessor.PRODUCTION_TO_HUMAN36:
            human36_points[j] = production_points[i]
        return human36_points[1:]

    @staticmethod
    def human3d_prod_to_human32(human_prod_np):
        """
        Expands 22 points to the original 32. Original ones all equal zeros.
        It is required for visualization utilities.

        Parameters
        ----------
        human_prod_np : np.ndarray of shape [num_prod_kp, 3]
            Points to expand. num_prod_kp = H3P6_2D_NUM-1

        Returns
        -------
        np.ndarray of shape [32, 3]
        """
        assert len(human_prod_np) == H3P6_2D_NUM-1
        assert len(human_prod_np[0]) == 3
        human_prod_np = np.asarray(human_prod_np)
        h32 = np.zeros((H36_NUM_POINTS, 3), dtype='float32')
        h32[HumanProcessor.INDICES3DTO32POINTS] = human_prod_np
        return h32

    @staticmethod
    def human3d16tohuman32(human16points):
        """
        NOTICE! DEPRECATED METHOD

        Expands 16 points to the original 32. Original ones all equal zeros.
        It is required for visualization utilities.

        Parameters
        ----------
        human16points : np.ndarray of shape [16, 3]
            Points to expand.

        Returns
        -------
        np.ndarray of shape [32, 3]
        """
        return HumanProcessor.human3d_prod_to_human32(human16points)

    @staticmethod
    def prodtohuman32(prod):
        """
        Expands 23 points to the original 32. Original ones all equal zeros.
        It is required for visualization utilities.

        Parameters
        ----------
        prod : np.ndarray of shape [23, d]
            Points to expand.

        Returns
        -------
        np.ndarray of shape [32, d]
        """
        assert len(prod) == H3P6_2D_NUM
        prod = np.asarray(prod)
        d = prod.shape[-1]
        h32 = np.zeros((H3P6_3D_NUM_V1, d), dtype='float32')
        for h32_indx, h23_indx in HumanProcessor.H32_TO_HPROD_3D:
            h32[h32_indx] = prod[h23_indx]
        return h32

    @staticmethod
    def to_production_format(human36_points):
        dim = human36_points.shape[-1]
        # The 3D skeleton does not include central hip point, it is always zero
        # Add hip size
        t = np.zeros((H3P6_2D_NUM, dim))
        t[1:] = human36_points
        human36_points = t
        production_points = np.zeros((NUM_PROD_POINTS, dim), dtype='float32')
        for prod_point_id, human36_point_id in HumanProcessor.PRODUCTION_TO_HUMAN36:
            production_points[prod_point_id] = human36_points[human36_point_id]
        return production_points

    @staticmethod
    def init_from_lib():
        file_path = os.path.abspath(__file__)
        dir_path = pathlib.Path(file_path).parent
        data_stats_dir = os.path.join(dir_path, CONVERT_STATS_3D)
        mean_path = os.path.join(data_stats_dir, MEAN_2D)
        assert os.path.isfile(mean_path), f"Could not find mean_2d.npy in {mean_path}."
        mean_2d = np.load(mean_path)

        std_path = os.path.join(data_stats_dir, STD_2D)
        assert os.path.isfile(std_path), f"Could not find std_2d.npy in {std_path}."
        std_2d = np.load(std_path)

        mean_path = os.path.join(data_stats_dir, MEAN_3D)
        assert os.path.isfile(mean_path), f"Could not find mean_3d.npy in {mean_path}."
        mean_3d = np.load(mean_path)

        std_path = os.path.join(data_stats_dir, STD_3D)
        assert os.path.isfile(std_path), f"Could not find std_3d.npy in {std_path}."
        std_3d = np.load(std_path)
        return HumanProcessor(mean_2d, std_2d, mean_3d, std_3d)

    def __init__(self, mean_2d, std_2d, mean_3d, std_3d):
        self.mean2d = mean_2d.reshape(-1).astype('float32')
        self.std2d = std_2d.reshape(-1).astype('float32')
        self.mean3d = mean_3d.reshape(-1).astype('float32')
        self.std3d = std_3d.reshape(-1).astype('float32')

    def norm2d(self, human):
        if isinstance(human, Human):
            human = human.to_np()[:, :2]
        else:
            human = np.asarray(human)

        return (human.reshape(-1) - self.mean2d) / self.std2d

    def denorm2d(self, human):
        return human.reshape(-1) * self.std2d + self.mean2d

    def norm3d(self, human):
        if isinstance(human, Human):
            human = human.to_np_from3d()[:, :3]
        else:
            human = np.asarray(human)

        return (human.reshape(-1) - self.mean3d) / self.std3d

    def denorm3d(self, human):
        return human.reshape(-1) * self.std3d + self.mean3d
