import numpy as np

from .one_euro_filter import OneEuroFilter
from petools.tools import Human


MODE_2D = '2d'
MODE_3D = '3d'


class OneEuroModule:
    """
    This correction module is based in 1 euro algorithm
    For mode details refer to: https://hal.inria.fr/hal-00670496/document
    """

    def __init__(self, freq=1.0, mincutoff=0.4, beta=0.001, dcutoff=0.7, mode=MODE_2D):
        self._euro_list = None
        self._freq = freq
        self._mincutoff = mincutoff
        self._beta = beta
        self._dcutoff = dcutoff
        self._mode = mode

    def __call__(self, human: Human) -> Human:
        if self._mode == MODE_2D:
            return self.__filter_2d(human)
        elif self._mode == MODE_3D:
            return self.__filter_3d(human)

        raise ValueError(f"Unknown type of filter mode, can be: {MODE_3D} or {MODE_3D},\nbut {self._mode} was given.")

    def __filter_2d(self, human: Human):
        # (N, 3)
        single_human = human.to_np()
        points_xy = np.zeros(single_human[:, :-1].shape).astype(np.float32)

        if self._euro_list is None:
            # 2 - X and Y axis
            self.__setup_filter(len(points_xy) * 2)

        for i in range(len(single_human)):
            # Its better to drop filtering value, if its disappear or pop-up often
            if single_human[i][-1] < 1e-3:
                self._euro_list[i * 2].reset_values()       # X
                self._euro_list[i * 2 + 1].reset_values()   # Y
            else:
                points_xy[i, 0] = self._euro_list[i * 2](single_human[i, 0])        # X
                points_xy[i, 1] = self._euro_list[i * 2 + 1](single_human[i, 1])    # Y

        return Human.from_array(np.concatenate([points_xy, single_human[:, 2:3]], axis=-1))

    def __filter_3d(self, human: Human):
        # (N, 4)
        single_human = human.to_np_from3d()
        points_xyz = np.zeros(single_human[:, :-1].shape).astype(np.float32)

        if self._euro_list is None:
            # 3 - X, Y and Z axis
            self.__setup_filter(len(points_xyz) * 3)

        for i in range(len(single_human)):
            # Its better to drop filtering value, if its disappear or pop-up often
            if single_human[i][-1] < 1e-3:
                self._euro_list[i * 3].reset_values()       # X
                self._euro_list[i * 3 + 1].reset_values()   # Y
                self._euro_list[i * 3 + 2].reset_values()   # Z
            else:
                points_xyz[i, 0] = self._euro_list[i * 3](single_human[i, 0])        # X
                points_xyz[i, 1] = self._euro_list[i * 3 + 1](single_human[i, 1])    # Y
                points_xyz[i, 2] = self._euro_list[i * 3 + 2](single_human[i, 2])    # Z

        return Human.from_array_3d(np.concatenate([points_xyz, single_human[:, 3:4]], axis=-1))

    def __setup_filter(self, num_elements):
        self._euro_list = [
            OneEuroFilter(self._freq, self._mincutoff, self._beta, self._dcutoff)
            for _ in range(num_elements)
        ]

