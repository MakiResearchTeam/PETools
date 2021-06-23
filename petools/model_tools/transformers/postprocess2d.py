import numpy as np

from . import HumanProcessor
from .core import DataProcessor
from ...tools import Human
from .utils import HIP_ID


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
        h, w = kwargs['source_resolution']
        coords2d *= h / 700
        # Add fake neck point because corrector does not regress it
        npoints, dim = coords2d.shape
        t = np.zeros((npoints + 1, dim), dtype='float32')
        t[:11] = coords2d[:11]
        t[12:] = coords2d[11:]
        coords2d = t
        coords2d = self.human_processor.to_production_format(coords2d)
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
