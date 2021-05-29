import numpy as np

from petools.tools.estimate_tools import Human


class SequenceBuffer:
    # Simple buffer to keep poses from previous frames
    def __init__(self, dim, seqlen=32):
        """
        A structure that records a window of `seqlen` humans.
        Used in tandem with transformer models as they require a context of `seqlen` humans.

        Parameters
        ----------
        dim : int
            Number of points each human contains.
        seqlen : int
            Sequence length.
        """
        self.dim = dim
        self.seqlen = seqlen
        self.token_sequence = np.zeros(shape=(1, self.seqlen, dim), dtype='float32')
        self.mask_sequence = np.zeros(shape=(1, self.seqlen), dtype='float32')
        self.absent_token = np.zeros(shape=dim, dtype='float32')

    @property
    def sequence(self):
        return self.token_sequence, self.mask_sequence

    def __call__(self, token: np.ndarray = None):
        assert token.shape[0] == self.dim, f'Token dimensionality must be {self.dim} but received {token.shape}'

        present = 1
        if token is None:
            token = self.absent_token
            present = 0

        # Shift previous humans one step past
        self.token_sequence[0, :-1] = self.token_sequence[0, 1:]
        self.token_sequence[0, -1] = token

        self.mask_sequence[0, :-1] = self.mask_sequence[0, 1:]
        self.mask_sequence[0, -1] = present
        return self.sequence

