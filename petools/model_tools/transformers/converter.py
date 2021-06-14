import tensorflow as tf
import numpy as np

from petools.core import Op, ProtobufModel
from .human_processor import HumanProcessor
from ...tools import Human
from .seq_buffer import SequenceBuffer
from .transformer import Transformer


class TransformerConverter(Op):
    def __init__(
            self,
            transformer: Transformer,
            human_processor: HumanProcessor,
            seq_len=32,
    ):
        """
        2d-3d converter.

        Parameters
        ----------
        transformer : Transformer
            The actual model.
        human_processor : HumanProcessor
            HumanProcessor object for data processing.
        seq_len : int
            Sequence length used by the transformer model.
        """
        self._transformer = transformer
        self._buffer = SequenceBuffer(dim=32, seqlen=seq_len)
        self._human_processor = human_processor

    def __call__(self, human: Human, **kwargs) -> Human:
        source_resolution = kwargs['source_resolution']
        human_ = human
        human = self.preprocess(human, source_resolution)
        coords3d = self.convert(human)
        coords3d = self.postprocess(coords3d, human_)
        human_.set_3d(coords3d)
        return human_

    # --- PREPROCESSING

    def preprocess(self, human: Human, source_resolution):
        human = human.to_np()[:, :2]
        human = self.shift_and_scale(human, source_resolution)
        human = self._human_processor.to_human36_format(human).reshape(-1)
        human = self._human_processor.norm2d(human)
        return human

    def shift_and_scale(self, human: np.ndarray, source_resolution):
        assert len(human.shape) == 2
        h, w = source_resolution
        human *= 700 / h
        # Shift the skeleton into the center of 1000x1000 square image
        selected_x = human[:, 0][human[:, 0] != 0]
        left_x = 0
        if len(selected_x) != 0:
            left_x = np.min(selected_x)
        right_x = np.max(human[:, 0])
        width = right_x - left_x
        center = left_x + width / 2
        if np.max(human[:, 0]) > 900:
            shift = center - 500
            human[:, 0] -= shift
        elif np.min(human[:, 0]) < 100:
            shift = 500 - center
            human[:, 0] += shift
        return human

    # --- CONVERTATION

    def convert(self, human):
        human_seq, mask_seq = self._buffer(human)
        output_seq = self._transformer.predict_poses(human_seq, mask_seq)[0]
        # Return the last pose
        return output_seq[0, -1]

    # --- POSTPROCESSING

    def postprocess(self, coords3d, human: Human):
        if coords3d.shape[0] == 48:
            denorm = lambda x: self._human_processor.denorm3d(x)
            dim = 3
        else:
            denorm = lambda x: self._human_processor.denorm2d(x)
            dim = 2
        coords3d = denorm(coords3d).reshape(-1, dim)
        coords3d = self._human_processor.to_production_format(coords3d)
        # Concatenate probabilities of converted points
        present_points = (coords3d[:, 0] != 0.0).astype('float32')
        p = np.expand_dims(human.to_np()[:, -1] * present_points, axis=-1)
        coords3d = np.concatenate([coords3d, p], axis=-1)
        return coords3d
