from petools.tools import Logging

from ..core import PairingProtocol, SIMMAT, PAIRING_INFO


class GreedyPairing(PairingProtocol):
    """
    Assumptions:
    - similarity value is a float;
    - the larger the value, the less similar features are.
    """
    def __init__(self, threshold: float):
        """
        Parameters
        ----------
        threshold : float
            Minimal similarity value between features to consider them close enough to pair.
        """
        self.threshold = threshold
        self.logger = Logging.get_logger(self.__class__.__name__)

    def find_minimum(self, similarity_mat: SIMMAT, used_human_inds: set):
        min_sim = 1e10
        mins_human_ind = None
        mins_feature_id = None
        for feature_id, sim_list in similarity_mat.items():
            for human_ind, sim_val in sim_list:
                if human_ind in used_human_inds:
                    continue
                # noinspection PyTypeChecker
                if sim_val < min_sim:
                    min_sim = sim_val
                    mins_human_ind = human_ind
                    mins_feature_id = feature_id

        if mins_human_ind is None or mins_feature_id is None:
            self.logger.debug('Minimum value was not found. '
                              f'similarity_mat={similarity_mat}, used_human_inds={used_human_inds}')

        self.logger.debug(
            f'Minimum sim_val={min_sim}, mins_human_ind={mins_human_ind}, mins_feature_id={mins_feature_id}'
        )
        return min_sim, mins_human_ind, mins_feature_id

    def pairing(self, similarity_mat: SIMMAT, **kwargs) -> PAIRING_INFO:
        self.logger.debug(
            f'Received similarity_mat={similarity_mat}, kwargs={kwargs}.'
        )
        similarity_mat = similarity_mat.copy()
        pairing_info = []
        used_human_inds = set()
        while len(similarity_mat) > 0:
            sim_val, human_ind, feature_id = self.find_minimum(similarity_mat, used_human_inds)
            if sim_val < self.threshold and human_ind is not None and feature_id is not None:
                # --- Found a pair
                self.logger.debug(
                    f'Adding pairing info: sim_val={sim_val}, feature_id={feature_id}, human_ind={human_ind}.'
                )
                pairing_info.append((feature_id, human_ind))
                used_human_inds.add(human_ind)
                self.logger.debug(f'Popping a mat row with feature_id={feature_id}.')
                similarity_mat.pop(feature_id)
                continue
            elif sim_val >= self.threshold and human_ind is not None and feature_id is not None:
                # --- Found a pair but its too dissimilar
                # The rest are too dissimilar, finish pairing
                print('WHAT THE FUCKING FUCK')
                self.logger.debug('The rest of humans are too dissimilar. Finishing pairing. '
                                  f'The remaining similarity_mat={similarity_mat}, '
                                  f'threshold={self.threshold}.')
                for feature_id in similarity_mat.keys():
                    pairing_info.append((feature_id, None))
                break

            if human_ind is None or feature_id is None:
                # Probably there are less humans than features meaning all humans were paired with a feature ID
                self.logger.debug(f'No minimum found. The remaining similarity_mat={similarity_mat}.')
                self.logger.debug(f'Probably there are less humans than features,'
                                  f' meaning all humans were paired with a feature ID. Finish pairing.')

                for feature_id in similarity_mat.keys():
                    pairing_info.append((feature_id, None))
                break

        self.logger.debug(f'Pairing info: {pairing_info}')
        return pairing_info
