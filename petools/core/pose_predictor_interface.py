import numpy as np
from abc import ABC, abstractmethod


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


if __name__ == '__main__':
    print(PosePredictorInterface.HUMANS)

