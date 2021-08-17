import numpy as np

from petools.tools import Human
from .pose_preprocessor import PosePreprocessor


class Preprocess2DPose:

    def __init__(self, pose_preprocessor: PosePreprocessor):
        """
        A class for preprocessing 2d data before feeding the converter transformer.

        Parameters
        ----------
        pose_preprocessor : HumanProcessor
        """
        self.pose_preprocessor = pose_preprocessor

    def __call__(self, human: Human, **kwargs):
        norm_human_np = self.pose_preprocessor.norm2d(human)
        return norm_human_np

