import numpy as np
import os
import pathlib

from petools.tools.estimate_tools import Human


class HumanProcessor:
    """
    This is a utility for normalizing/denormalizing data from Human3.6M dataset.
    """

    PRODUCTION_TO_HUMAN36 = [
        # LOWER BODY
        # middle hip
        [22, 0],
        # right hip
        [11, 1],
        # left hip
        [10, 4],

        # right knee
        [13, 2],
        # left knee
        [12, 5],

        # right foot
        [15, 3],
        # left foot
        [14, 6],

        # UPPER BODY
        # center
        [0, 7],
        [1, 8],
        # left shoulder
        [4, 11],
        # right shoulder
        [5, 14],

        # neck
        [2, 9],
        # head
        [3, 10],

        # HANDS
        # left elbow
        [6, 12],
        # right elbow
        [7, 15],

        # left wrist
        [8, 13],
        # right wrist
        [9, 16]
    ]

    def to_human36_format(self, production_points):
        human36_points = np.zeros((17, 2), dtype='float32')
        for i, j in HumanProcessor.PRODUCTION_TO_HUMAN36:
            human36_points[j] = production_points[i]
        return human36_points[self._select_2d]

    # noinspection PyMethodMayBeStatic
    def to_production_format(self, human36_points):
        dim = human36_points.shape[-1]
        # The 3D skeleton does not include central hip point, it is always zero
        t = np.zeros((17, dim))
        t[1:] = human36_points
        human36_points = t
        production_points = np.zeros((23, dim), dtype='float32')
        for prod_point_id, human36_point_id in HumanProcessor.PRODUCTION_TO_HUMAN36:
            production_points[prod_point_id] = human36_points[human36_point_id]
        return production_points

    @staticmethod
    def init_from_lib():
        file_path = os.path.abspath(__file__)
        dir_path = pathlib.Path(file_path).parent
        data_stats_dir = os.path.join(dir_path, '3d_converter_stats')
        mean_path = os.path.join(data_stats_dir, 'mean_2d.npy')
        assert os.path.isfile(mean_path), f"Could not find mean_2d.npy in {mean_path}."
        mean_2d = np.load(mean_path)

        std_path = os.path.join(data_stats_dir, 'std_2d.npy')
        assert os.path.isfile(std_path), f"Could not find std_2d.npy in {std_path}."
        std_2d = np.load(std_path)

        mean_path = os.path.join(data_stats_dir, 'mean_3d.npy')
        assert os.path.isfile(mean_path), f"Could not find mean_3d.npy in {mean_path}."
        mean_3d = np.load(mean_path)

        std_path = os.path.join(data_stats_dir, 'std_3d.npy')
        assert os.path.isfile(std_path), f"Could not find std_3d.npy in {std_path}."
        std_3d = np.load(std_path)
        return HumanProcessor(mean_2d, std_2d, mean_3d, std_3d)

    def __init__(self, mean_2d, std_2d, mean_3d, std_3d):
        self.mean2d = mean_2d.reshape(-1).astype('float32')
        self.std2d = std_2d.reshape(-1).astype('float32')
        self.mean3d = mean_3d.reshape(-1).astype('float32')
        self.std3d = std_3d.reshape(-1).astype('float32')

        # The input to the network does not include neck point
        self._select_2d = [True] * 17
        self._select_2d[9] = False

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