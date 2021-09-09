from abc import abstractmethod
from .feature_extractor import HumanRepresentation


class Similarity:
    """
    Performs a comparison of two feature vectors and returns a value
    describing how similar the two feature vectors are.
    """
    @abstractmethod
    def __call__(self, f1: HumanRepresentation, f2: HumanRepresentation, **kwargs) -> object:
        pass

    def reset(self):
        pass

