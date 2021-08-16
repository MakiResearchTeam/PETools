import tensorflow.compat.v1 as tf

from petools.core import ProtobufModel
from ..utils import H36_2DPOINTS_DIM_FLAT


class Transformer(ProtobufModel):

    def __init__(self, protobuf_path: str, seq_len=32, session: tf.Session = None):
        super().__init__(
            protobuf_path, input_map={
                'input': ('float32', [1, seq_len, H36_2DPOINTS_DIM_FLAT], 'input_'),
                'mask': ('float32', [1, seq_len], 'mask_')
            },
            output_tensors=['Identity:0'],
            session=session
        )

        self._input_sequence = self.input_map['input']
        self._input_mask = self.input_map['mask']

    def predict_poses(self, input_seq, input_mask):
        return super().predict(
            {
                self._input_sequence: input_seq,
                self._input_mask: input_mask
            }
        )
