from typing import List, Tuple

from .typing import FEATURE_ID, HUMAN_IND, SIMMAT, PAIRING_INFO


class PairingProtocol:
    """
    Performs pairing of humans to already registered features' IDs.
    """
    def pairing(self, similarity_mat: SIMMAT, **kwargs) -> PAIRING_INFO:
        """
        Parameters
        ----------
        similarity_mat : dict
            Dictionary with the following contents: {feature_id : [(human_ind, similiarity_value)]}

        Returns
        -------
        PAIRING_INFO
            List with the following contents: [(feature_id, human_ind)].
        """
        pass
