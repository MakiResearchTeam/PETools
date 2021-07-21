import numpy as np


class HumanCleaner:
    """
    Remove humans with low threshold

    """

    def __init__(self, min_visible=2):
        """

        Parameters
        ----------
        min_visible : int
            Minimum values of visible keypoints,
            if input human class have more visible keypoints then its good,
            otherwise human class will be ignored (i.e. deleted)

        """
        self._min_visible = min_visible

    def __call__(self, humans: list) -> list:
        """
        Clean input list of Human classes according to number of visible keypoints

        Returns
        -------
        list
            List of good human classes, with more than `min_visible` keypoints

        """
        good_humans = []
        for human in humans:
            # TODO: Force Human class calculate number of visible parts
            num_visible = np.sum(human.to_np()[:, -1] > 1e-3)
            if num_visible > self._min_visible:
                good_humans.append(human)

        return good_humans

