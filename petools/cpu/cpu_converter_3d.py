import tensorflow.compat.v1 as tf
import numpy as np

from ..core import Converter3D


class CpuConverter3D(Converter3D):
    def __init__(self, tflite_path, mean_2d, std_2d, mean_3d, std_3d):
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
        super().__init__(mean_2d, std_2d, mean_3d, std_3d)

        self._interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # For inference
        self._input_tensor = self._input_details[0]['index']
        self._input_shape = self._input_details[0]['shape']

        self._output_tensor = self._output_details[0]['index']

    def _predict(self, normalized_points):
        self._interpreter.set_tensor(self._input_tensor, normalized_points)
        self._interpreter.invoke()
        return self._interpreter.get_tensor(self._output_tensor)

