from typing import List
import numpy as np

from petools.tools import Human


class HumanCleaner:
    """
    Remove humans with low threshold

    """

    def __init__(self, min_visible=2, threshold_visibility=0.2):
        """

        Parameters
        ----------
        min_visible : int
            Minimum values of visible keypoints,
            if input human class have more visible keypoints then its good,
            otherwise human class will be ignored (i.e. deleted)
        threshold_visibility : float
            If point bigger than `threshold_visibility` then this point will be visible,
            otherwise this point will be as not visible
            By default equal to 0.2

        """
        self._min_visible = min_visible
        self._threshold_visibility = threshold_visibility

    def __call__(self, humans: List[Human]) -> List[Human]:
        """
        Clean input list of Human classes according to number of visible keypoints

        Returns
        -------
        list
            List of good human classes, with more than `min_visible` keypoints

        """
        good_humans = []
        for human in humans:
            num_visible = np.sum(human.np[:, -1] > self._threshold_visibility)
            if num_visible > self._min_visible:
                good_humans.append(human)

        return good_humans

