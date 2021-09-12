from typing import List, Tuple, Union
from numpy import linalg as LA

from .. import HumanRepresentation
from ..core import Similarity, REPRESENTATION_ID, SIMMAT
from .representation import CustomRepresentation
from ..common import SimpleSimilarity
from .gaussian_measure import CustomGauss


class CustomSimilarity(SimpleSimilarity):
    def __init__(self, distance=0.1):
        super().__init__(sim_measure=CustomGauss())
        self.distance = distance

    def compute_similarity_matrix(self, registered_representations: List[Tuple[REPRESENTATION_ID, CustomRepresentation]],
                                  new_representations: List[CustomRepresentation], **kwargs) -> Union[SIMMAT, None]:
        n, m = len(registered_representations), len(new_representations)
        # --- Compute distances between representations and turn off decay if needed
        for id1, reg_repr1 in registered_representations:
            for id2, reg_repr2 in registered_representations:
                if id1 == id2:
                    continue
                dist = LA.norm(reg_repr1.xy - reg_repr2.xy)
                if self.debug_enabled:
                    self.debug_log(f'Distance between reprs with id1={id1} and id2={id2} is dist={dist}.')
                if dist < self.distance:
                    reg_repr1.xy_weights.start_decay()
                    if self.debug_enabled:
                        self.debug_log(f'Started decay on representation with id={id1}.')
                    break

        return super().compute_similarity_matrix(registered_representations, new_representations, **kwargs)


