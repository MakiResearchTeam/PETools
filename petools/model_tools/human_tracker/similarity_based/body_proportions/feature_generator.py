import numpy as np

from ..core import FeatureExtractor
from .feature_generator import FeatureGenerator
from .representation import CustomRepresentation, XYDecayingWeights


class FExtractor(FeatureExtractor):
    def __init__(self, indices: np.ndarray, conlist: list, mean: np.ndarray = None, std: np.ndarray = None, n_points=23,
                 features_std_scale=3.0, features_weight_scale=1.0, xy_std_scale=0.1, xy_weight_scale=10.0,
                 p_threshold=0.5):
        """
        Custom feature extractor used in human tracking pipeline.

        Parameters
        ----------
        indices : np.ndarray
            List of indices of features to take.
        conlist : list
            List of pairs of indices denoting "limbs". Used by feature generator.
        mean : np.ndarray
            Used to normalize the generated features.
        std : np.ndarray
            Used to normalize the generated features.
        n_points : int
            Used for dummy check in the feature generator.
        features_std_scale : float
            Determines the value of the features' std when computing Gaussian kernel. See gaussian_measure.py.
        features_weight_scale : float
            Determines the value of the features' weights when computing weighted sum. See gaussian_measure.py.
        xy_std_scale : float
            Same as `features_std_scale`, but for the mean point.
        xy_weight_scale : float
            Same as `features_weight_scale`, but for the mean point.
        p_threshold : float
            Determines the minimum confidence value for a point to be considered during mean point computation.
        """
        super().__init__()
        self.fg = FeatureGenerator(conlist, n_points, 1e-6)
        self.mean = mean
        self.std = std
        self.p_threshold = p_threshold
        # Perform fake generation to obtain info about the output shapes
        temp_f = self.fg.generate_features(np.random.randn(n_points, 2).astype('float32'))
        self.low_diag_indices = np.tril_indices(temp_f.shape[0], k=-1)
        self.indices = indices[:-1]
        self.n_used_features = temp_f[self.low_diag_indices][self.indices]

        self.init_buffers(features_std_scale, features_weight_scale, xy_std_scale, xy_weight_scale)

    # noinspection PyAttributeOutsideInit
    def init_buffers(self, std_scale, weights_scale, xy_weight_std, xy_weight_scale):
        self.std_features = np.ones(self.n_used_features, dtype='float32') * std_scale
        self.feature_weights = np.ones(self.n_used_features, dtype='float32') * weights_scale
        self.xy_weights = np.ones(1, dtype='float32') * xy_weight_scale
        self.xy_std = np.ones(1, dtype='float32') * xy_weight_std

    def compute_mean_point(self, human_np, **kwargs):
        select = human_np[:, -1] > self.p_threshold
        points = human_np[:, :-1][select]
        mean_point = points.mean(axis=0)
        image_size = kwargs['image_size']
        h, w = image_size
        mean_point[0] /= w
        mean_point[1] /= h
        return mean_point

    def compute_body_features(self, human):
        human = human.to_np()
        f = self.fg.generate_features(human)
        f = f[self.low_diag_indices].reshape(-1)
        f = f[self.indices]
        f = np.clip(f, 0, 3)
        if self.mean and self.std:
            f = (f - self.mean) / self.std
        return f

    def __call__(self, human, **kwargs):
        f = self.compute_body_features(human)
        xy = self.compute_mean_point(human, **kwargs)
        xy_weights = XYDecayingWeights(xy.shape, 80, init=self.xy_weights)
        return CustomRepresentation(f, self.std_features, xy, self.xy_std, self.feature_weights, xy_weights)
