from abc import abstractmethod
from typing import List, Tuple, Union

from .typing import SIMMAT, REPRESENTATION_ID
from .feature_extractor import HumanRepresentation


class Similarity:
    """
    Performs a comparison of two feature vectors and returns a value
    describing how similar the two feature vectors are.
    """
    @abstractmethod
    def compute_similarity_matrix(
            self,
            registered_representations: List[Tuple[REPRESENTATION_ID, HumanRepresentation]],
            new_representations: List[HumanRepresentation],
            **kwargs
    ) -> Union[SIMMAT, None]:
        pass

    def reset(self):
        pass

