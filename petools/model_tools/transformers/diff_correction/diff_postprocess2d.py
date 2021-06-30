import numpy as np

from petools.model_tools.transformers import HumanProcessor, SequenceBuffer
from petools.model_tools.transformers.core import DataProcessor
from petools.tools import Human
from petools.model_tools.transformers.utils import HIP_ID


class DiffPostprocess2D(DataProcessor):
    def __init__(self, human_processor: HumanProcessor, buffer: SequenceBuffer, tolerance=0.5):
        """
        A class for preprocessing 2d data before feeding the corrector transformer.

        Parameters
        ----------
        human_processor : HumanProcessor
        buffer : SequenceBuffer
        tolerance : float
            A float in range [0, +inf]
        """
        self.human_processor = human_processor
        self.buffer = buffer
        self.tolerance = tolerance

    def __call__(self, transformed_data: np.ndarray, source_human: Human, **kwargs):
        coords2d = self.denorm_data(transformed_data, **kwargs)
        # Add fake neck point because corrector does not regress it
        coords2d = self.add_fake_neck_point(coords2d)
        coords2d = self.human_processor.to_production_format(coords2d)
        prev_human, diff = self.get_prev_human(source_human)
        mask = (np.abs(diff - coords2d) / diff) > self.tolerance
        # Move the hip point to its original position and nullify points that were previously absent
        mask = (coords2d[:, 0] == 0.0).astype('float32')
        coords2d += source_human.to_np()[HIP_ID:HIP_ID+1, :2]
        coords2d *= 1 - np.expand_dims(mask, axis=-1)
        # Replace only those points that are not presents in coords2d
        coords2d = coords2d + source_human.to_np()[:, :2] * np.expand_dims(mask, axis=-1)
        # Concatenate probabilities of converted points
        p = source_human.to_np()[:, -1:]
        coords2d = np.concatenate([coords2d, p], axis=-1)

        human = Human.from_array(coords2d)
        human.id = source_human.id
        return human

    def denorm_data(self, transformed_data, **kwargs):
        # denormalize the transformer's answer
        coords2d = (transformed_data * self.human_processor.std2d).reshape(-1, 2)
        h, w = kwargs['source_resolution']
        coords2d *= h / 700
        return coords2d

    def add_fake_neck_point(self, coords):
        npoints, dim = coords.shape
        t = np.zeros((npoints + 1, dim), dtype='float32')
        t[:11] = coords[:11]
        t[12:] = coords[11:]
        return t

    def get_prev_human(self, human) -> (np.ndarray, np.ndarray):
        """
        Saves current human into the buffer and returns the human from the previous frame.

        Parameters
        ----------
        human : Human
            Current human.

        Returns
        -------
        np.ndarray
            Previous human.
        np.ndarray
            Difference between current human and the last human.
        """
        human = human.to_np().reshape(-1)
        seq = self.buffer(human)
        return seq[-2], seq[-1] - seq[-2]
