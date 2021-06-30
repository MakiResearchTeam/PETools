import numpy as np

from petools.model_tools.transformers.core import DataProcessor, SequenceBuffer
from petools.model_tools.transformers.human_processor import HumanProcessor
from petools.tools.estimate_tools import Human
from petools.model_tools.transformers.preprocess3d import Preprocess3D


class DiffPreprocess2D(Preprocess3D):
    def __init__(self, human_processor: HumanProcessor, buffer: SequenceBuffer):
        """
        A class for preprocessing 2d data before feeding the converter transformer.

        Parameters
        ----------
        human_processor : HumanProcessor
        """
        super(DiffPreprocess2D, self).__init__(human_processor)
        self.humans_buffer = buffer
        self.flag_first = True

    def __call__(self, human: Human, skip_hip=False, **kwargs):
        source_resolution = kwargs['source_resolution']
        human_np = human.to_np()
        human, p = human_np[:, :2], human_np[:, 2:]
        p = np.concatenate([p] * 2, axis=-1)
        human = self.shift_and_scale(human, source_resolution)
        human = self.human_processor.to_human36_format(human)
        p = self.human_processor.to_human36_format(p)
        human = self.center_around_zero_point(human, p)
        if skip_hip:
            # Skip hip
            human = human[1:]
        human = human.reshape(-1)
        human = self.human_processor.norm2d(human)

        seq = self.humans_buffer(human)
        diff = seq[1:] - seq[:-1]
        if self.flag_first:
            self.flag_first = False
            return np.zeros_like(diff)
        return diff[-1]
