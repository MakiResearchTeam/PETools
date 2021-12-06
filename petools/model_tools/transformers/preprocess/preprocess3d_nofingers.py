import numpy as np

from petools.model_tools.transformers import HumanProcessor
from petools.model_tools.transformers.core import DataProcessor
from petools.tools import Human


class Preprocess3DNF(DataProcessor):
    def __init__(self, human_processor: HumanProcessor):
        """
        A class for preprocessing 2d data before feeding the converter transformer.

        Parameters
        ----------
        human_processor : HumanProcessor
        """
        self.human_processor = human_processor

    def __call__(self, human: Human, skip_hip=False, **kwargs):
        # Copy array in order to keep original array safe
        human_np = human.to_np(copy_if_cached=True)
        human, p = human_np[:, :2], human_np[:, 2:]
        p = np.concatenate([p]*2, axis=-1)
        human = self.human_processor.to_human36_format(human)
        p     = self.human_processor.to_human36_format(p)
        human = self.center_around_zero_point(human, p)
        # Skip hip
        human = human[1:]
        human = np.delete(human, [3, 7, 14, 15, 19, 20], axis=0)
        human = human.reshape(-1)
        #human = self.human_processor.norm2d(human)
        return human

    def center_around_zero_point(self, human, probabilities):
        mask = (probabilities > 1e-3).astype('float32', copy=False)
        # Mask out absent points
        return (human - human[0:1]) * mask
