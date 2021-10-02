from typing import List, Tuple, Union
from abc import abstractmethod

from petools.model_tools.human_tracker.similarity_based import HumanRepresentation
from petools.model_tools.human_tracker.similarity_based.core import Similarity, SIMMAT
from petools.tools.logging import LoggedEntity


class SimilarityMeasure(LoggedEntity):
    @abstractmethod
    def __call__(self, f1: HumanRepresentation, f2: HumanRepresentation, **kwargs):
        pass


class SimpleSimilarity(Similarity):
    def __init__(self, sim_measure):
        super().__init__()
        self.sim_measure = sim_measure

    def compute_similarity_matrix(
            self,
            registered_representations: List[Tuple[int, HumanRepresentation]],
            new_representations: List[HumanRepresentation],
            **kwargs
    ) -> Union[SIMMAT, None]:
        """
        Computes a "similarity matrix" containing similarity values between humans' representations
        and already registered representations.
        Returns None if there are no representations registered at the moment.

        Parameters
        ----------
        registered_representations : list
            List of registered representations.
        new_representations : list
            List of newly computed representations.

        Returns
        -------
        similarity_mat : dict
            Dictionary with the following contents: {feature_id : [(human_ind, similarity_value)]}
        """
        if len(registered_representations) == 0:
            return None

        similarity_mat = dict()  # {feature_id: (human_ind, similarity_value)}
        for id, registered_representation in registered_representations:
            similarities = []
            similarity_mat[id] = similarities
            for human_ind, human_representation in enumerate(new_representations):
                if self.debug_enabled:
                    self.debug_log(
                        f'Computing similarity between (registered_id, representation_ind): ({id}, {human_ind})'
                    )
                sim_val = self.sim_measure(registered_representation, human_representation, **kwargs)
                similarities.append((human_ind, sim_val))
        return similarity_mat
