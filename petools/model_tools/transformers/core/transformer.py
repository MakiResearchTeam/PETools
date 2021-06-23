import tensorflow as tf

from petools.core import ProtobufModel
from ..utils import H36_2DPOINTS_DIM_FLAT


class Transformer(ProtobufModel):
    def __init__(self, protobuf_path: str):
        super().__init__(
            protobuf_path, input_tensor_names=['input:0', 'mask:0'],
            output_tensors_names=['Identity:0'],
        )

    def predict_poses(self, input_seq, input_mask):
        return super().predict(input_seq, input_mask)
