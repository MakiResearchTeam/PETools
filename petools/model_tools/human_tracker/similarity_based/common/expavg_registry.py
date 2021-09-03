import logging
from dataclasses import dataclass
from logging import getLogger

from ..core import FeatureRegistry, FEATURE_ID, FeatureIterator


@dataclass
class FeaturesHolder:
    features: object
    id: int
    # Number of frames the owner of this feature was absent (for example, human walked out of the frame)
    n_absent: int = 0


class Iter(FeatureIterator):
    def __init__(self, holder_dict):
        self.holder_iter = holder_dict.items()
        self.counter = 0
        self.n = len(holder_dict)

    def __next__(self):
        if self.counter >= self.n:
            return
        id, holder = next(self.holder_iter)
        features = holder.features
        self.counter += 1
        return id, features


class ExpAvgRegistry(FeatureRegistry):
    def __init__(self, alpha: float = 0.9, expiration_time: int = 10):
        """
        Parameters
        ----------
        alpha : float
            The features being updated using exponential averaging: f = f_old * (1 - alpha) + f_new * alpha.
        expiration_time : int
            How much frame a human must be absent to remove it from the registry.
        """
        super().__init__(alpha=alpha, expiration_time=expiration_time)
        assert 0.0 < alpha <= 1.0, f'Alpha must be in (0, 1], but received alpha={alpha}.'
        self.alpha = alpha
        self.expiration_time = expiration_time
        self.id_counter = 0

    def register_features(self, features) -> FEATURE_ID:
        holder = FeaturesHolder(features, self.id_counter)
        self._register_features(self.id_counter, holder)
        self.id_counter += 1
        return holder.id

    def update_features(self, id: FEATURE_ID, features):
        holder = self.get_feature(id)
        if features is None:
            holder.n_absent += 1
        else:
            holder.n_absent = 0

        # Update feature values using exponential averaging
        holder.features = holder.features * (1.0 - self.alpha) + features * self.alpha

    # noinspection PyTypeChecker
    def __iter__(self) -> FeatureIterator:
        return Iter(self.registry)

    def update_state(self):
        # Remove expired feature holders
        for id, holder in self.registry.items():
            if holder.n_absent >= self.expiration_time:
                self.registry.pop(id)
                self.logger.info(f'Feature holder with id={id} has expired. Removing.')

