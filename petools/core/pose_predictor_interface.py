import numpy as np
from abc import ABC, abstractmethod

from petools.core.utils import CONNECT_KP
from petools.tools import draw_skeleton


class PosePredictorInterface(ABC):
    SCALE = 8
    NUM_KEYPOINTS = 23

    HUMANS = 'humans'
    TIME = 'time'

    @abstractmethod
    def predict(self, image: np.ndarray):
        """
        Estimate poses on single image

        Parameters
        ----------
        image : np.ndarray
            Input image, with shape (H, W, 3): H - Height, W - Width (H and W can have any values)
            For mose models - input image must be in bgr order

        Returns
        -------
        dict
            Single predictions as dict object contains of:
            {
                PosePredictorInterface.HUMANS: [
                        [
                            [h1_x_1, h1_y_1, h1_v_1],
                            [h1_x_2, h1_y_2, h1_v_2],
                            ...
                            [h1_x_n, h1_y_n, h1_v_n],
                        ],

                        [
                            [h2_x_1, h2_y_1, h2_v_1],
                            [h2_x_2, h2_y_2, h2_v_2],
                            ...
                            [h2_x_n, h2_y_n, h2_v_n],
                        ]

                        ...
                        ...

                        [
                            [hN_x_1, hN_y_1, hN_v_1],
                            [hN_x_2, hN_y_2, hN_v_2],
                            ...
                            [hN_x_n, hN_y_n, hN_v_n],
                        ]
                ],
                PosePredictorInterface.TIME: some_float_number
            }
            Where PosePredictorInterface.HUMANS and PosePredictorInterface.TIME - are strings ('humans' and 'time')
        """
        pass

    @staticmethod
    def draw(image: np.ndarray, predictions: dict, color=(255, 0, 0), thick=3):
        """
        Draw skeletons from `preidctions` on certain `image`
        With parameters such as color and thick of the line

        Parameters
        ----------
        image : np.ndarray
            The image on which detection was performed
        predictions : dict
            Prediction on `image` from this class and method `predict`
        color : tuple
            Color of the line,
            By default equal to (255, 0, 0) - i.e. red line
        thick : int
            Thick of the line, by default equal to 3, in most cases this value is enough

        Returns
        -------
        np.ndarray
            Image with skeletons on it

        """
        predictions_humans = predictions[PosePredictorInterface.HUMANS]
        humans = [list(single_h.values()) for single_h in predictions_humans]
        return draw_skeleton(image.copy(), humans, connect_indexes=CONNECT_KP, color=color, thickness=thick)


if __name__ == '__main__':
    print(PosePredictorInterface.HUMANS)

