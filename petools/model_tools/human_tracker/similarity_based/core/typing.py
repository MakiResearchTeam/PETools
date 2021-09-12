from typing import Union, Dict, List, Tuple

REPRESENTATION_ID = int
# It is index of a human, not an ID.
HUMAN_IND = Union[int, None]
SIMILARITY_VALUE = object
# Similarity matrix
SIMMAT = Dict[REPRESENTATION_ID, List[Tuple[HUMAN_IND, SIMILARITY_VALUE]]]
PAIRING_INFO = List[Tuple[REPRESENTATION_ID, HUMAN_IND]]
