import numpy as np

from . import HumanProcessor
from .core import DataProcessor
from ...tools import Human


class Postprocess2D(DataProcessor):
    def __init__(self, human_processor: HumanProcessor):
        """
        A class for preprocessing 2d data before feeding the converter transformer.

        Parameters
        ----------
        human_processor : HumanProcessor
        """
        self.human_processor = human_processor

    def __call__(self, transformed_data: np.ndarray, source_human: Human, **kwargs):
        coords2d = self.human_processor.denorm2d(transformed_data).reshape(-1, 2)
        coords2d = self.human_processor.to_production_format(coords2d)
        # Replace only those points that are presents in coords2d
        present_points = (coords2d[:, 0] == 0.0).astype('float32')
        coords2d = coords2d + source_human.to_np()[:, :2] * np.expand_dims(present_points, axis=-1)
        # Concatenate probabilities of converted points
        p = source_human.to_np()[:, -1:]
        coords2d = np.concatenate([coords2d, p], axis=-1)

        human = Human.from_array(coords2d)
        human.id = source_human.id
        return human
