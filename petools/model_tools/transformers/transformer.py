import tensorflow as tf

from petools.core import ProtobufModel


class Transformer(ProtobufModel):
    def __init__(self, protobuf_path: str, seq_len=32, session: tf.Session = None):
        self._input_sequence = tf.placeholder(dtype='float32', shape=[1, seq_len, 32], name='input_')
        self._input_mask = tf.placeholder(dtype='float32', shape=[1, seq_len], name='mask_')
        super().__init__(
            protobuf_path, input_map={
                'input': self._input_sequence,
                'mask': self._input_mask
            },
            output_tensors=['Identity:0'],
            session=session
        )

    def predict_poses(self, input_seq, input_mask):
        return super().predict(
            {
                self._input_sequence: input_seq,
                self._input_mask: input_mask
            }
        )
