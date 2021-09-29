from typing import List, Tuple, Union
from numpy import linalg as LA

from .. import HumanRepresentation
from ..core import Similarity, REPRESENTATION_ID, SIMMAT
from .representation import FancyRepresentation
from ..common import SimpleSimilarity
from .gaussian_measure import GaussianMeasure


class FancySimilarity(SimpleSimilarity):
    def __init__(self, distance=0.075, min_height_ratio=1.125):
        super().__init__(sim_measure=GaussianMeasure())
        self.distance = distance
        self.min_height_ratio = min_height_ratio

    def compute_similarity_matrix(self, registered_representations: List[Tuple[REPRESENTATION_ID, FancyRepresentation]],
                                  new_representations: List[FancyRepresentation], **kwargs) -> Union[SIMMAT, None]:
        # --- Compute distances between representations and turn on decay if needed
        for id1, reg_repr1 in registered_representations:
            for id2, reg_repr2 in registered_representations:
                if id1 == id2:
                    continue

                dist = LA.norm(reg_repr1.xy - reg_repr2.xy)

                min_h = min(reg_repr1.height, reg_repr2.height)
                max_h = max(reg_repr1.height, reg_repr2.height)
                ratio = max_h / min_h

                if self.debug_enabled:
                    self.debug_log(f'Distance between reprs with id1={id1} and id2={id2} is dist={dist}.')

                if dist < self.distance and ratio < self.min_height_ratio:
                    reg_repr1.xy_weights.start_decay()
                    if self.debug_enabled:
                        self.debug_log(f'Started decay on representation with id={id1}.')
                    break

        return super().compute_similarity_matrix(registered_representations, new_representations, **kwargs)


