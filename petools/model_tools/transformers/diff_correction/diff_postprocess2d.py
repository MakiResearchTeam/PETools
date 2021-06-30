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
            A positive float that defines the tolerance interval for the difference between corrector's
            predictions and actual values. If the difference too large, the corresponding points are being masked.
        """
        self.human_processor = human_processor
        self.buffer = buffer
        self.tolerance = tolerance

    def __call__(self, transformed_data: np.ndarray, source_human: Human, **kwargs):
        predicted_diff = self.denorm_data(transformed_data, **kwargs)
        # Add fake neck point because corrector does not regress it
        predicted_diff = self.add_fake_neck_point(predicted_diff)
        predicted_diff = self.human_processor.to_production_format(predicted_diff)
        mask = self.generate_mask(source_human, predicted_diff)
        source_data = source_human.to_np()
        p, coords = source_data[:, -1:], source_data[:, :2]
        # Update mask
        mask = (p > 0).astype('float32') * mask
        # Update human data (coords, probs)
        p = p * mask
        coords = coords * mask
        new_human_data = np.concatenate([coords, p], axis=-1)

        human = Human.from_array(new_human_data)
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

    def generate_mask(self, cur_human, predicted_diff):
        """
        Generates a binary mask for masking the erroneous points.

        Parameters
        ----------
        cur_human : Human

        Returns
        -------
        np.ndarray of shape [n_points, 1]
            Binary mask.
        """
        cur_human = cur_human.to_np().reshape(-1)
        _, diff = self.get_prev_human(cur_human)
        mask = (np.abs(diff - predicted_diff) / predicted_diff) < self.tolerance
        mask = mask.astype('float32')
        mask = mask.reshape(-1, 2)
        mask = mask[:, 0] * mask[:, 1]
        return np.expand_dims(mask, -1)

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
