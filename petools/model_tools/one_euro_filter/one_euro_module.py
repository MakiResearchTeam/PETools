from .one_euro_filter import OneEuroFilter
from petools.tools import Human


MODE_2D = '2d'
MODE_3D = '3d'


class OneEuroModule:
    """
    This correction module is based in 1 euro algorithm
    For mode details refer to: https://hal.inria.fr/hal-00670496/document
    """

    __slots__ = ('_euro_list', '_freq', '_mincutoff', '_beta', '_dcutoff', '_mode', '_prev_points')

    THR = 0.2

    def __init__(self, freq=1.0, mincutoff=0.4, beta=0.001, dcutoff=0.7, mode=MODE_2D):
        self._euro_list = None
        self._freq = freq
        self._mincutoff = mincutoff
        self._beta = beta
        self._dcutoff = dcutoff
        self._mode = mode
        # Some parameters as buffer
        # In order to speed up work of this class
        self._prev_points = None

    def __call__(self, human: Human) -> Human:
        if self._mode == MODE_2D:
            return self.__filter_2d(human)
        elif self._mode == MODE_3D:
            return self.__filter_3d(human)

        raise ValueError(f"Unknown type of filter mode, can be: {MODE_3D} or {MODE_3D},\nbut {self._mode} was given.")

    def __filter_2d(self, human: Human):
        num_points = len(human.np)

        if self._euro_list is None:
            # 2 because of - X and Y axis
            self.__setup_filter(num_points * 2)

        for i in range(num_points):
            # Its better to drop filtering value, if its disappear or pop-up often
            if human.np[i, -1] < OneEuroModule.THR:
                self._euro_list[i * 2].reset_values()       # X
                self._euro_list[i * 2 + 1].reset_values()   # Y
            else:
                human.np[i, 0] = self._euro_list[i * 2](human.np[i, 0])        # X
                human.np[i, 1] = self._euro_list[i * 2 + 1](human.np[i, 1])    # Y

        return human

    def __filter_3d(self, human: Human):
        num_points = len(human.np3d)

        if self._euro_list is None:
            # 3 - X, Y and Z axis
            self.__setup_filter(num_points * 3)

        for i in range(num_points):
            # Its better to drop filtering value, if its disappear or pop-up often
            if human.np3d[i, -1] < 1e-3:
                self._euro_list[i * 3].reset_values()       # X
                self._euro_list[i * 3 + 1].reset_values()   # Y
                self._euro_list[i * 3 + 2].reset_values()   # Z
            else:
                human.np3d[i, 0] = self._euro_list[i * 3](human.np3d[i, 0])        # X
                human.np3d[i, 1] = self._euro_list[i * 3 + 1](human.np3d[i, 1])    # Y
                human.np3d[i, 2] = self._euro_list[i * 3 + 2](human.np3d[i, 2])    # Z

        return human

    def __setup_filter(self, num_elements):
        self._euro_list = [
            OneEuroFilter(self._freq, self._mincutoff, self._beta, self._dcutoff)
            for _ in range(num_elements)
        ]

