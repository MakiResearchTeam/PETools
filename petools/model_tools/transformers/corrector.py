import numpy as np

from .converter import TransformerConverter
from petools.tools.estimate_tools import Human


class TransformerCorrector(TransformerConverter):
    """
    This function is different from that of a TransformerConverter in a way
    that it does not scale the human by its height, but the source image width in order
    to preserve information about the absolute position of a human.
    """
    def shift_and_scale(self, human: np.ndarray, source_resolution):
        assert len(human.shape) == 2
        h, w = source_resolution
        print('before norm', human)
        human *= 800 / w
        human[:, -1] -= 100
        print('after norm', human)
        return human

    def __call__(self, human: Human, **kwargs) -> Human:
        source_resolution = kwargs['source_resolution']
        human_ = human
        human = self.preprocess(human, source_resolution)
        coords2d = self.convert(human)
        coords2d = self.postprocess(coords2d, human_, source_resolution)
        human = Human.from_array(coords2d)
        human.id = human_.id
        return human

    def postprocess(self, coords2d, human: Human, source_resolution):
        if coords2d.shape[0] == 48:
            denorm = lambda x: self._human_processor.denorm3d(x)
            dim = 3
        else:
            denorm = lambda x: self._human_processor.denorm2d(x)
            dim = 2
        coords2d = denorm(coords2d).reshape(-1, dim)
        coords2d = self._human_processor.to_production_format(coords2d)
        # Restore the width scale
        coords2d *= source_resolution[1] / 1000
        # Concatenate probabilities of converted points
        present_points = (coords2d[:, 0] == 0.0).astype('float32')
        print(present_points)
        print('coords2d before', coords2d)
        coords2d = coords2d + human.to_np()[:, :2] * np.expand_dims(present_points, axis=-1)
        print('coords2d after', coords2d)
        p = human.to_np()[:, -1:]
        coords2d = np.concatenate([coords2d, p], axis=-1)
        return coords2d
