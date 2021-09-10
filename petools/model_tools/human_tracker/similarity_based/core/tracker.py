from typing import List, Union
import logging

from petools.tools import Logging, log
from petools.core import Tracker
from petools.tools import Human
from .feature_extractor import FeatureExtractor
from .feature_registry import RepresentationRegistry
from .similarity import Similarity
from .pairing_protocol import PairingProtocol
from .typing import SIMMAT


class SimilarityBasedTracker(Tracker):
    def __init__(
            self, feature_extractor: FeatureExtractor,
            similarity: Similarity,
            representation_registry: RepresentationRegistry,
            pairing_protocol: PairingProtocol
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.similarity = similarity
        self.representation_registry = representation_registry
        self.pairing_protocol = pairing_protocol
        self.frame_number = 0

    @log
    def __call__(self, humans: List[Human], **kwargs):
        if self.debug_enabled:
            self.debug_log(f'--- TRACKING ON FRAME {self.frame_number}.\n')
            self.debug_log('Received humans:')
            for human in humans:
                self.debug_log(human.get_human_info())
            self.debug_log('')
        # --- Extract features
        if self.debug_enabled:
            self.debug_log(f'--- EXTRACTING FEATURES')
        human_representations = []
        for human in humans:
            human_representations.append(self.feature_extractor(human, **kwargs))

        if self.debug_enabled:
            self.debug_log(f'Computed representations for humans:')
            for feature in human_representations:
                self.debug_log(feature)
            self.debug_log('')

        # --- Compute similarity and assign IDs
        if self.debug_enabled:
            self.debug_log(f'--- COMPUTING SIMILARITY MATRIX')
        similarity_mat = self.similarity.compute_similarity_matrix(
            self.representation_registry.representations,
            human_representations,
            **kwargs
        )
        # --- Perform pairing
        if self.debug_enabled:
            self.debug_log(f'--- PERFORMING REPRESENTATIONS PAIRING')
        paired_humans = self.id_pairing(similarity_mat, humans, human_representations, **kwargs)

        # If some humans were not paired with an existing ID, register them and pair with a new ID
        if self.debug_enabled:
            self.debug_log(f'--- REGISTERING UNPAIRED REPRESENTATIONS')
        for human_ind, is_paired in paired_humans.items():
            if is_paired:
                continue
            # Human with such features was not paired. Register the feature vector
            if self.debug_enabled:
                self.debug_log(f'Human (ind={human_ind}) with the following representation was not paired:')
                self.debug_log(human_representations[human_ind])

            representation_id = self.representation_registry.register_representation(human_representations[human_ind])
            humans[human_ind].id = representation_id

            if self.debug_enabled:
                self.debug_log(f"This human representation (ind={human_ind}) was registered with an id={representation_id}.\n")

        # Let the registry know that no new information about this frame will come.
        self.representation_registry.update_state()

        if self.debug_enabled:
            self.debug_log(f'--- FINISHED TRACKING ON FRAME {self.frame_number}.\n\n\n')

        self.frame_number += 1
        return humans

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
                self.representation_registry.update_representation(feature_id, None)
                continue

            # Perform ID pairing.
            humans[human_ind].id = feature_id
            paired_humans[human_ind] = True
            self.representation_registry.update_representation(feature_id, human_features[human_ind])

        return paired_humans

    def reset(self):
        self.representation_registry.reset()
        self.feature_extractor.reset()
        self.similarity.reset()
        self.pairing_protocol.reset()
        self.frame_number = 0
