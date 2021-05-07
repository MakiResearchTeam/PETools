import tensorflow as tf
import numpy as np

from ..core import Converter3D
from petools.tools.utils.tf_tools import load_graph_def
from petools.tools.estimate_tools import Human


class GpuConverter3D(Converter3D):
    def __init__(self, pb_path, mean_2d, std_2d, mean_3d, std_3d, input_name='input', output_name='Identity',
                 session=None):
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
        super().__init__(mean_2d, std_2d, mean_3d, std_3d)

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
        )[0]
        if session is None:
            session = tf.Session()
        self._sess = session

    def _predict(self, normalized_points):
        return self._sess.run(
            self._3d_coords,
            feed_dict={self._in_x: normalized_points}
        )