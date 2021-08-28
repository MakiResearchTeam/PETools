import numpy as np

from petools.tools import Human
from .pose_preprocessor import PosePreprocessor
from petools.model_tools.transformers.utils import NUM_PROD_POINTS
from .feature_generator import FeatureGenerator
from ..utils import NUM_C, conlist


class Preprocess2DPose:

    def __init__(self, pose_preprocessor: PosePreprocessor):
        """
        A class for preprocessing 2d data before feeding the converter transformer.

        Parameters
        ----------
        pose_preprocessor : HumanProcessor
        """
        self.pose_preprocessor = pose_preprocessor
        self.fg = FeatureGenerator(np.array(conlist, dtype=np.int32), NUM_PROD_POINTS)
        self.features = np.empty(shape=(self.fg.n_triples * NUM_C), dtype=np.float32)
        print('features shape: ', self.features.shape)

    def __call__(self, human: Human, **kwargs):
        human = human.to_np(copy_if_cached=True)[:, :-1]
        print('human: ', human.shape)
        self.fg.generate_features(human, features=self.features)
        norm_human_np = self.pose_preprocessor.norm2d(self.features)
        return norm_human_np

