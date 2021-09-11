import numpy as np

from .representation import CustomRepresentation
from ..common import SimilarityMeasure


class CustomGauss(SimilarityMeasure):
    def kernel(self, x, kernel_mean, kernel_std):
        exp_arg = (kernel_mean - x) / kernel_std
        kernel_val = np.exp(-np.square(exp_arg))
        return kernel_val

    def __call__(self, f1: CustomRepresentation, f2: CustomRepresentation, **kwargs) -> np.float:
        """
        Computes similarity, using gaussian kernel.

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

        # TODO
        # Compute l2 distance between mean points and then pass it through the kernel
        kernel_val_xy = self.kernel(f2.xy, f1.xy, f1.std_xy)
        if self.debug_enabled:
            self.debug_log(f'kernel_val_xy: {kernel_val_xy}')
        #print('kernel_val_features', kernel_val_features)
        #print('kernel_val_xy', kernel_val_xy)
        # --- Compute weighted average
        num = np.dot(kernel_val_features, f1.features_weights) + np.dot(kernel_val_xy, f1.xy_weights.weights)
        if self.debug_enabled:
            self.debug_log(f'num: {num}')
        den = np.sum(f1.features_weights) + np.sum(f1.xy_weights.weights)
        if self.debug_enabled:
            self.debug_log(f'den: {den}')

        #print('num', num)
        #print('den', den)
        return num / den
