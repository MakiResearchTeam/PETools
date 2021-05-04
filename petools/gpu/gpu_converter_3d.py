import tensorflow as tf
import numpy as np

from ..core import Converter3D
from petools.tools.utils.tf_tools import load_graph_def
from petools.tools.estimate_tools import Human


class GpuConverter3D(Converter3D):
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

    def __init__(self, pb_path, mean, std, input_name='input', output_name='Identity', session=None):
        """
        2d-3d converter.

        Parameters
        ----------
        pb_path : str
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
        self._mean = mean.reshape(1, -1).astype('float32')
        self._std = std.reshape(1, -1).astype('float32')

        self._graph_def = load_graph_def(pb_path)
        self._in_x = tf.placeholder(dtype=tf.float32, shape=[1, 32], name='in_x')
        self._3d_coords = tf.import_graph_def(
            self._graph_def,
            input_map={
                input_name: self._in_x
            },
            return_elements=[
                output_name
            ]
        )
        if session is None:
            session = tf.Session()
        self._sess = session

        self._points_buffer = np.zeros(1, 17, 2).astype('float32')
        self._points_buffer_nn = np.zeros(1, 16, 2).astype('float32')

    def predict(self, points: np.ndarray):
        self.fill_points_buffer(points)
        return self._sess.run(
            self._3d_coords,
            feed_dict={self._in_x: self.normalize_points_buffer()}
        )

    def fill_points_buffer(self, points: np.ndarray):
        for i, j in GpuConverter3D.PRODUCTION_TO_HUMAN36:
            self._points_buffer[j] = points[i]
        self._points_buffer_nn[:, :14] = self._points_buffer[:, :14]
        self._points_buffer_nn[:, 14:] = self._points_buffer[:, 15:]

    def normalize_points_buffer(self):
        return (self._points_buffer_nn.reshape(1, -1) - self._mean) / self._std

    def __call__(self, skeletons, source_resolution):
        h, w = source_resolution
        skeletons_3d = []
        for skeleton in skeletons:
            skeleton = skeleton.to_np()[:, :2]
            skeleton[:, 0] /= w * 1000
            skeleton[:, 1] /= h * 1000
            skeleton_3d = self.predict(skeleton).reshape(16, 3)
            skeletons_3d.append(Human.from_array(skeleton_3d))
        return skeletons_3d
