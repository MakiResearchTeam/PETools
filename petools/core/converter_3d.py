import numpy as np
from abc import abstractmethod


class Converter3D:
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

    def __init__(self, mean_2d, std_2d, mean_3d, std_3d):
        """
        2d-3d converter.

        Parameters
        ----------
        tflite_path : str
            Path to the protobuf file with model's graph.
        mean_2d : np.ndarray of shape [32]
            Mean statistics for data normalization.
        std_2d : np.ndarray of shape [32]
            Std statistics for data normalization.
        mean_3d : np.ndarray of shape [48]
            Mean statistics for predictions denormalization.
        std_3d : np.ndarray of shape [48]
            Std statistics for predictions denormalization.
        """
        self._mean_2d = mean_2d.reshape(1, -1).astype('float32')
        self._std_2d = std_2d.reshape(1, -1).astype('float32')
        self._mean_3d = mean_3d.reshape(1, -1).astype('float32')
        self._std_3d = std_3d.reshape(1, -1).astype('float32')

        # TODO remove these idiot buffers
        self._points_buffer = np.zeros((1, 17, 2)).astype('float32')
        self._points_buffer_nn = np.zeros((1, 16, 2)).astype('float32')

    @abstractmethod
    def _predict(self, normalized_points):
        pass

    def predict(self, points: np.ndarray):
        self.fill_points_buffer(points)
        pred = self._predict(self.normalize_points_buffer())
        return self.denormalize(pred)

    def fill_points_buffer(self, points: np.ndarray):
        for i, j in Converter3D.PRODUCTION_TO_HUMAN36:
            self._points_buffer[0, j] = points[i]
        self._points_buffer_nn[:, :14] = self._points_buffer[:, :14]
        self._points_buffer_nn[:, 14:] = self._points_buffer[:, 15:]

    def normalize_points_buffer(self):
        return (self._points_buffer_nn.reshape(1, -1) - self._mean_2d) / self._std_2d

    def denormalize(self, prediction):
        return prediction * self._std_3d + self._mean_3d

    def __call__(self, skeletons, source_resolution):
        h, w = source_resolution
        skeletons_3d = []
        for skeleton in skeletons:
            skeleton_2d = skeleton.to_np()[:, :2]
            skeleton_2d *= 1000 / h
            # Shift the skeleton into the center of 1000x1000 square image
            selected_x = skeleton_2d[:, 0][skeleton_2d[:, 0] != 0]
            left_x = 0
            if len(selected_x) != 0:
                left_x = np.min(selected_x)
            right_x = np.max(skeleton_2d[:, 0])
            width = right_x - left_x
            center = left_x + width / 2
            if np.max(skeleton_2d[:, 0]) > 900:
                shift = center - 500
                skeleton_2d[:, 0] -= shift
            elif np.min(skeleton_2d[:, 0]) < 100:
                shift = 500 - center
                skeleton_2d[:, 0] += shift

            skeleton_3d = self.predict(skeleton_2d).reshape(16, 3)
            skeletons_3d.append(self.pack_skeleton(skeleton_3d, skeleton.to_np()))
        return skeletons_3d

    def pack_skeleton(self, skeleton_3d: np.ndarray, skeleton_2d: np.ndarray):
        new_skeleton = np.zeros((17, 3))
        new_skeleton[:14] = skeleton_3d[:14]
        new_skeleton[15:] = skeleton_3d[14:]

        out_dict = {}
        for prod_point_id, human36_point_id in Converter3D.PRODUCTION_TO_HUMAN36:
            out_dict[f'p{prod_point_id}'] = new_skeleton[human36_point_id].tolist()
            # Add confidence of the corresponding point
            out_dict[f'p{prod_point_id}'].append(float(skeleton_2d[prod_point_id][-1]))

        return out_dict
