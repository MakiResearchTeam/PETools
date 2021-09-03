from abc import abstractmethod


class Similarity:
    """
    Performs a comparison of two feature vectors and returns a value
    describing how similar the two feature vectors are.
    """
    @abstractmethod
    def __call__(self, f1, f2, **kwargs) -> object:
        pass

