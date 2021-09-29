import numpy as np

from .representation import FancyRepresentation
from ..common import SimilarityMeasure


class GaussianMeasure(SimilarityMeasure):
    def kernel(self, x, kernel_mean, kernel_std):
        exp_arg = (kernel_mean - x) / kernel_std
        kernel_val = np.exp(-np.square(exp_arg))
        return kernel_val

    def __call__(self, f1: FancyRepresentation, f2: FancyRepresentation, **kwargs) -> np.float:
        """
        Computes similarity, using a gaussian kernel.

        Parameters
        ----------
        f1 : GaussianRepresentation
            This must be a representation from the registry. Contains kernel parameters.
        f2 : GaussianRepresentation
            This must be a newly computed human representation.

        Returns
        -------
        float
            Similarity value. The higher the value, the more similar the representations are.
        """
        # --- Compute kernel for features
        kernel_val_features = self.kernel(f2.features, f1.features, f1.std_features)
        if self.debug_enabled:
            self.debug_log(f'kernel_val_features: {kernel_val_features}')

        # Compute l2 distance between mean points and then pass it through the kernel
        dist = np.linalg.norm(f1.xy - f2.xy)
        kernel_val_xy = np.exp(-np.square(dist / f1.std_xy))
        if self.debug_enabled:
            self.debug_log(f'kernel_val_xy: {kernel_val_xy}')

        # --- Compute kernel for height
        kernel_val_height = self.kernel(f2.height, f1.height, f1.std_height)
        if self.debug_enabled:
            self.debug_log(f'kernel_val_height: {kernel_val_features}')

        # --- Compute weighted average
        num = np.dot(kernel_val_features, f1.features_weights) + np.dot(kernel_val_xy, f1.xy_weights.weights)
        num += kernel_val_height * f1.height_weight
        den = np.sum(f1.features_weights) + np.sum(f1.xy_weights.weights) + f1.height_weight
        if self.debug_enabled:
            self.debug_log(f'numerator value: {num}')
            self.debug_log(f'denominator value: {den}')
        return num / den
