from petools.tools import Logging

from ..core import PairingProtocol, SIMMAT, PAIRING_INFO


class GreedyPairing(PairingProtocol):
    """
    Assumptions:
    - similarity value is a float;
    - the larger the value, the less similar features are.
    """

    def __init__(self, threshold: float = 0.1, comparator=lambda a, b: a > b, worst_similarity_value=0.01):
        """
        Parameters
        ----------
        threshold : float
            Minimal similarity value between features to consider them close enough to pair.
        """
        super().__init__()
        self.threshold = threshold
        self.comparator = comparator
        self.worst_similarity_value = worst_similarity_value

    def find_best_candidate(self, similarity_mat: SIMMAT, used_human_inds: set):
        best_sim = self.worst_similarity_value
        best_human_ind = None
        best_feature_id = None
        for feature_id, sim_list in similarity_mat.items():
            for human_ind, sim_val in sim_list:
                if human_ind in used_human_inds:
                    continue
                # noinspection PyTypeChecker
                if self.comparator(sim_val, best_sim):
                    best_sim = sim_val
                    best_human_ind = human_ind
                    best_feature_id = feature_id

        if best_human_ind is None or best_feature_id is None:
            if self.debug_enabled:
                self.debug_log('Best value was not found. '
                               f'similarity_mat={similarity_mat}, used_human_inds={used_human_inds}')

        if self.debug_enabled:
            self.debug_log(
                f'Best sim_val={best_sim}, best_human_ind={best_human_ind}, best_feature_id={best_feature_id}'
            )
        return best_sim, best_human_ind, best_feature_id

    def pairing(self, similarity_mat: SIMMAT, **kwargs) -> PAIRING_INFO:
        if self.debug_enabled:
            self.debug_log(
                f'Received kwargs={kwargs} and the following similarity_mat:'
            )
            for id, sim_info in similarity_mat.items():
                self.debug_log(f'{id} / {sim_info}')

        similarity_mat = similarity_mat.copy()
        pairing_info = []
        used_human_inds = set()
        while len(similarity_mat) > 0:
            sim_val, human_ind, feature_id = self.find_best_candidate(similarity_mat, used_human_inds)
            fulfills_threshold = self.comparator(sim_val, self.threshold)
            if fulfills_threshold and human_ind is not None and feature_id is not None:
                # --- Found a pair
                if self.debug_enabled:
                    self.debug_log(
                        f'Adding pairing info: sim_val={sim_val}, feature_id={feature_id}, human_ind={human_ind}.'
                    )
                pairing_info.append((feature_id, human_ind))
                used_human_inds.add(human_ind)
                if self.debug_enabled:
                    self.debug_log(f'Popping a mat row with feature_id={feature_id}.')
                similarity_mat.pop(feature_id)
                continue
            elif not fulfills_threshold and human_ind is not None and feature_id is not None:
                # --- Found a pair but its too dissimilar
                # The rest are too dissimilar, finish pairing
                if self.debug_enabled:
                    self.debug_log('The rest of humans are too dissimilar. Finishing pairing. '
                                   f'The remaining similarity_mat={similarity_mat}, '
                                   f'threshold={self.threshold}.')
                for feature_id in similarity_mat.keys():
                    pairing_info.append((feature_id, None))
                break

            if human_ind is None or feature_id is None:
                # Probably there are less humans than features meaning all humans were paired with a feature ID
                if self.debug_enabled:
                    self.debug_log(f'No best candidate was found. The remaining similarity_mat={similarity_mat}.')
                    self.debug_log(f'Probably there are less humans than features,'
                                   f' meaning all humans were paired with a feature ID. Finish pairing.')

                for feature_id in similarity_mat.keys():
                    pairing_info.append((feature_id, None))
                break

        if self.debug_enabled:
            self.debug_log(f'Pairing info: {pairing_info}')
        return pairing_info
