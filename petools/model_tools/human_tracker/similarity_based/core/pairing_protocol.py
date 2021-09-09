from typing import List, Tuple
from abc import abstractmethod

from .typing import REPRESENTATION_ID, HUMAN_IND, SIMMAT, PAIRING_INFO


class PairingProtocol:
    """
    Performs pairing of humans to already registered features' IDs.
    """
    @abstractmethod
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

    def reset(self):
        pass
