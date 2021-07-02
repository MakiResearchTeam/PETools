import numpy as np

from .preprocess3d import Preprocess3D


def dist(x, y):
    return np.linalg.norm(x - y)


# Their functionality is the same, but it is useful to keep them distinct anyway
class Preprocess2D(Preprocess3D):
    def __call__(self, *args, **kwargs):
        human, _ = super().__call__(*args, **kwargs)
        human_ = human.reshape(-1, 2)
        scale = (dist(human_[7], human_[8]) + dist(human_[2], human_[3])) / 2
        human /= scale
        return human, scale

