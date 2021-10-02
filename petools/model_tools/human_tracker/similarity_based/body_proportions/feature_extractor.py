import numpy as np
import os
import pathlib

from ..core import FeatureExtractor
from .feature_generator import FeatureGenerator
from .representation import FancyRepresentation, XYDecayingWeights
from petools.model_tools.pose_classifier.process_data.feature_generator import FeatureGenerator as CFeatureGenerator
from petools.model_tools.pose_classifier.utils import conlist as CF_CONLIST
from petools.model_tools.pose_classifier.utils import INPUT_SHAPE


class FExtractor(FeatureExtractor):
    """
    Computes various features that help to identify an individual.
    """
    @staticmethod
    def init_from_lib(n_points=23,
                      features_std_scale=1.0, features_weight_scale=27.0, xy_std_scale=0.1, xy_weight_scale=10.0,
                      p_threshold=0.1, decay_steps=80, upper_points_inds=(4, 5, 2), lower_points_inds=(10, 11, 22)):
        """
        See the constructor for the parameters description.
        """
        file_path = os.path.abspath(__file__)
        dir_path = pathlib.Path(file_path).parent
        stats_dir = os.path.join(dir_path, 'data_statistics')
        mean = np.load(os.path.join(stats_dir, 'f_mean.npy'))
        std = np.load(os.path.join(stats_dir, 'f_std.npy'))
        indices = np.load(os.path.join(stats_dir, 'feature_indices.npy'))
        from .data_statistics.constants import conlist
        return FExtractor(
            indices=indices, conlist=conlist, mean=mean, std=std,
            n_points=n_points,
            features_std_scale=features_std_scale, features_weight_scale=features_weight_scale,
            xy_std_scale=xy_std_scale, xy_weight_scale=xy_weight_scale,
            p_threshold=p_threshold, decay_steps=decay_steps,
            upper_points_inds=upper_points_inds, lower_points_inds=lower_points_inds
        )

    def __init__(self, indices: np.ndarray, conlist: list, mean: np.ndarray = None, std: np.ndarray = None, n_points=23,
                 features_std_scale=3.0, features_weight_scale=1.0, xy_std_scale=0.1, xy_weight_scale=10.0,
                 p_threshold=0.1, decay_steps=80, upper_points_inds=(4, 5, 2), lower_points_inds=(10, 11, 22)):
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
        decay_steps : int
            Determines for how long the xy weights will decay.
        upper_points_inds : tuple
            Indices of the upper keypoints. Used to calculate height of the upper body.
        lower_points_inds : tuple
            Indices of the lower keypoints. Used to calculate height of the upper body.
        """
        super().__init__()
        self.fg = FeatureGenerator(conlist, n_points, 1e-6)
        self.mean = mean
        self.std = std
        self.p_threshold = p_threshold
        self.decay_steps = decay_steps
        # Perform fake generation to obtain info about the output shapes
        temp_f = self.fg.generate_features(np.random.randn(n_points, 2).astype('float32'))
        self.low_diag_indices = np.tril_indices(temp_f.shape[0], k=-1)
        self.indices = indices[:-1]
        self.n_used_features = len(temp_f[self.low_diag_indices][self.indices]) + INPUT_SHAPE
        self.upper_points_inds = upper_points_inds
        self.lower_points_inds = lower_points_inds
        self.classifier_fextractor = CFeatureGenerator(connectivity_list=np.array(CF_CONLIST), n_points=n_points)
        self.features = np.empty(shape=INPUT_SHAPE, dtype=np.float32)
        self.init_buffers(features_std_scale, features_weight_scale, xy_std_scale, xy_weight_scale)

    # noinspection PyAttributeOutsideInit
    def init_buffers(self, std_scale, weights_scale, xy_weight_std, xy_weight_scale):
        self.std_features = np.ones(self.n_used_features, dtype='float32') * std_scale
        self.feature_weights = np.ones(self.n_used_features, dtype='float32')
        self.feature_weights[:self.n_used_features - INPUT_SHAPE] *= 2.5
        self.feature_weights[self.n_used_features - INPUT_SHAPE:] /= 2.5
        self.feature_weights *= weights_scale / self.feature_weights.sum()
        self.xy_weights = np.ones(1, dtype='float32') * xy_weight_scale
        self.xy_std = np.ones(1, dtype='float32') * xy_weight_std

    def compute_mean_point(self, human: np.ndarray, **kwargs):
        select = human[:, -1] > self.p_threshold
        points = human[:, :-1][select]
        if len(points) == 0:
            # Return negative values. In this case the xy coords
            # of the corresponding registered representation won't be updated
            return np.asarray([-1, -1], dtype='float32')

        mean_point = points.mean(axis=0)
        image_size = kwargs['image_size']
        h, w = image_size
        mean_point[0] /= w
        mean_point[1] /= h
        return mean_point

    def compute_height(self, human: np.ndarray, **kwargs):
        image_size = kwargs['image_size']
        h, w = image_size
        upper_h = None
        for ind in self.upper_points_inds:
            if human[ind, 2] > self.p_threshold:
                upper_h = human[ind, 1]
        if upper_h is None:
            upper_h = 0.0

        lower_h = None
        for ind in self.lower_points_inds:
            if human[ind, 2] > self.p_threshold:
                lower_h = human[ind, 1]

        if lower_h is None:
            lower_h = h
        return (lower_h - upper_h) / h

    def compute_body_features(self, human: np.ndarray):
        f = self.fg.generate_features(human)
        f = f[self.low_diag_indices].reshape(-1)
        f = f[self.indices]
        f = np.clip(f, 0, 3)
        if self.mean and self.std:
            f = (f - self.mean) / self.std
        return f

    def __call__(self, human, **kwargs):
        human = human.to_np()
        f = self.compute_body_features(human)

        self.classifier_fextractor.generate_features(human[:, :-1], self.features)
        f = np.concatenate([f, self.features])

        xy = self.compute_mean_point(human, **kwargs)
        xy_weights = XYDecayingWeights(xy.shape, self.decay_steps, init=self.xy_weights)

        h = self.compute_height(human, **kwargs)

        return FancyRepresentation(
            f, self.std_features, self.feature_weights,
            xy, self.xy_std, xy_weights,
            h
        )
