from typing import Tuple

from .typing import FEATURE_ID


class FeatureIterator:
    """
    Iterates over already registered features.
    """
    def __next__(self) -> Tuple[FEATURE_ID, object]:
        pass


class FeatureRegistry:
    """
    Represents a buffer that holds feature vectors for registered (known) humans.
    It encapsulates how features are being processed from frame to frame (if such processing is needed, for example,
    exponential averaging among frames).
    """
    def register_features(self, features) -> FEATURE_ID:
        pass

    def update_features(self, id: FEATURE_ID, features):
        pass

    def has_features(self) -> bool:
        pass

    def __iter__(self) -> FeatureIterator:
        pass
