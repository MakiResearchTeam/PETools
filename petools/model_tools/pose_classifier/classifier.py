import tensorflow.compat.v1 as tf

from petools.core import ProtobufModel
from petools.model_tools.transformers.utils import H3P6_2D_NUM

INPUT_SHAPE = 55


class Classifier(ProtobufModel):
    """
    Classify given points as certain pose class

    """

    def __init__(
            self, protobuf_path: str, session: tf.Session = None, input_dims: int = INPUT_SHAPE):
        super().__init__(
            protobuf_path,
            input_map={'input': ('float32', [1, input_dims], 'input')},
            output_tensors=['Identity:0'],
            session=session
        )

        self._input_keypoints = self.input_map['input']

    def classify_poses(self, input_keypoints):
        return super().predict(
            {
                self._input_keypoints: input_keypoints
            }
        )

