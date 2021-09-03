from typing import Union, Dict, List, Tuple

FEATURE_ID = int
# It is index of a human, not an ID.
HUMAN_IND = Union[int, None]
SIMILARITY_VALUE = object
# Similarity matrix
SIMMAT = Dict[FEATURE_ID, List[Tuple[HUMAN_IND, object]]]
PAIRING_INFO = List[Tuple[FEATURE_ID, HUMAN_IND]]
