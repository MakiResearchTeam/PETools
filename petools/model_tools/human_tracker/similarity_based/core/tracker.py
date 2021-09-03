from typing import List, Union

from petools.core import Tracker
from petools.tools import Human
from .feature_extractor import FeatureExtractor
from .feature_registry import FeatureRegistry
from .similarity import Similarity
from .pairing_protocol import PairingProtocol
from .typing import SIMMAT


class SimilarityBasedTracker(Tracker):
    def __init__(
            self, feature_extractor: FeatureExtractor,
            similarity: Similarity,
            feature_registry: FeatureRegistry,
            pairing_protocol: PairingProtocol
    ):
        self.feature_extractor = feature_extractor
        self.similarity = similarity
        self.feature_registry = feature_registry
        self.pairing_protocol = pairing_protocol

    def __call__(self, humans: List[Human], **kwargs):
        # --- Extract features
        human_features = []
        for human in humans:
            human_features.append(self.feature_extractor(human, **kwargs))

        # Compute similarity and assign IDs
        similarity_mat = self.compute_simmat(human_features, **kwargs)
        paired_humans = self.id_pairing(similarity_mat, humans, human_features, **kwargs)

        # If some humans were not paired with an existing ID, register them and pair with a new ID
        for human_ind, is_paired in paired_humans.items():
            if is_paired:
                continue
            feature_id = self.feature_registry.register_features(human_features[human_ind])
            humans[human_ind].id = feature_id

        return humans

    def compute_simmat(self, human_features, **kwargs) -> Union[SIMMAT, None]:
        """
        Computes a "similarity matrix" containing similarity values between humans' and already stored features.
        Returns None if there are no features stored at the moment.

        Parameters
        ----------
        human_features : list
            List of features computed for the humans.

        Returns
        -------
        similarity_mat : dict
            Dictionary with the following contents: {feature_id : [(human_ind, similarity_value)]}
        """
        if self.feature_registry.has_features():
            return None

        similarity_mat = dict()  # {feature_id: (human_ind, similarity_value)}
        for feature_id, registered_feature in self.feature_registry:
            similarities = []
            similarity_mat[feature_id] = similarities
            for human_ind, human_feature in enumerate(human_features):
                sim_val = self.similarity(registered_feature, human_feature, **kwargs)
                similarities.append((human_ind, sim_val))
        return similarity_mat

    def id_pairing(self, similarity_mat: Union[SIMMAT, None], humans: List[Human], human_features: list, **kwargs):
        """
        Pairs humans with features' IDs according to the provided pairing protocol.

        Parameters
        ----------
        similarity_mat : dict
            Dictionary with the following contents: {feature_id : [(human_ind, similarity_value)]}.
        humans : List[Human]
            Humans to be assigned IDs.
        human_features : list
            List of features for the humans to be tracked. Used for updating the feature registry,
        Returns
        -------
        paired_humans : dict
            Shows whether the human with the corresponding index was paired with an existing ID.
        """
        # Shows which humans were paired with an ID.
        paired_humans = dict([(ind, False) for ind in range(len(humans))])
        if similarity_mat is None:
            return paired_humans

        pairing_info = self.pairing_protocol.pairing(similarity_mat, **kwargs)  # [(feature_id, human_ind)]
        for feature_id, human_ind in pairing_info:
            # Feature vector with the corresponding ID was not paired with any of the humans.
            if human_ind is None:
                # Let the registry know that the human with the corresponding ID is absent.
                self.feature_registry.update_features(feature_id, None)
                continue

            # Perform ID pairing.
            humans[human_ind].id = feature_id
            paired_humans[human_ind] = True
            self.feature_registry.update_features(feature_id, human_features[human_ind])

        return paired_humans
