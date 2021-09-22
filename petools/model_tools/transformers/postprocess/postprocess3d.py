import numpy as np

from petools.model_tools.transformers import HumanProcessor
from petools.model_tools.transformers.core import DataProcessor
from petools.tools import Human


class Postprocess3D(DataProcessor):
    def __init__(self, human_processor: HumanProcessor):
        """
        A class for preprocessing 2d data before feeding the converter transformer.

        Parameters
        ----------
        human_processor : HumanProcessor
        """
        self.human_processor = human_processor

    def __call__(self, transformed_data: np.ndarray, source_human: Human, **kwargs):
        coords3d = self.human_processor.denorm3d(transformed_data).reshape(-1, 3)
        coords3d = self.human_processor.to_production_format(coords3d)
        # Concatenate probabilities of converted points
        present_points = (coords3d[:, 0] != 0.0).astype('float32', copy=False)
        p = np.expand_dims(source_human.np[:, -1] * present_points, axis=-1)
        coords3d = np.concatenate([coords3d, p], axis=-1)

        source_human.set_3d(coords3d)
        return source_human
