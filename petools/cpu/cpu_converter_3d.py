import tensorflow as tf
import numpy as np

from ..core import Converter3D
from petools.tools.utils.tf_tools import load_graph_def
from petools.tools.estimate_tools import Human


class CpuConverter3D(Converter3D):
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

    def __init__(self, tflite_path, mean_2d, std_2d, mean_3d, std_3d):
        """
        2d-3d converter.

        Parameters
        ----------
        tflite_path : str
            Path to the protobuf file with model's graph.
        mean : np.ndarray of shape [32]
            Mean statistics for data normalization.
        std : np.ndarray of shape [32]
            Std statistics for data normalization.
        input_name : str
            Input tensor name.
        output_name : str
            Output tensor name.
        session : tf.Session
            THe session object to run the model.
        """
        self._mean_2d = mean_2d.reshape(1, -1).astype('float32')
        self._std_2d = std_2d.reshape(1, -1).astype('float32')
        self._mean_3d = mean_3d.reshape(1, -1).astype('float32')
        self._std_3d = std_3d.reshape(1, -1).astype('float32')

        self._interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # For inference
        self._input_tensor = self._input_details[0]['index']
        self._input_shape = self._input_details[0]['shape']

        self._output_tensor = self._output_details[0]['index']

        self._points_buffer = np.zeros((1, 17, 2)).astype('float32')
        self._points_buffer_nn = np.zeros((1, 16, 2)).astype('float32')

    def predict(self, points: np.ndarray):
        self.fill_points_buffer(points)
        self._interpreter.set_tensor(self._input_tensor, self._points_buffer_nn.reshape(1, -1))
        self._interpreter.invoke()
        pred = self._interpreter.get_tensor(self._output_tensor)
        return self.denormalize(pred)

    def fill_points_buffer(self, points: np.ndarray):
        for i, j in CpuConverter3D.PRODUCTION_TO_HUMAN36:
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
            skeleton = skeleton.to_np()[:, :2]
            skeleton[:, 0] /= w * 1000
            skeleton[:, 1] /= h * 1000
            skeleton_3d = self.predict(skeleton).reshape(16, 3)
            skeletons_3d.append(skeleton_3d.tolist())
        return skeletons_3d
