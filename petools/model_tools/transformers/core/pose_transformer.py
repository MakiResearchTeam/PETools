from .transformer import Transformer
from .seq_buffer import SequenceBuffer
from petools.tools import Human
from .data_processor import DataProcessor


import time


class PoseTransformer:
    def __init__(
            self,
            transformer: Transformer,
            seq_buffer: SequenceBuffer,
            preprocess: DataProcessor,
            postprocess: DataProcessor
    ):
        """
        Generic class for instance transforming a pose into another pose (possibly with another dimensionality).

        Parameters
        ----------
        transformer : Transformer
            The actual model.
        human_processor : HumanProcessor
            HumanProcessor object for data processing.
        seq_len : int
            Sequence length used by the transformer model.
        """
        self.transformer = transformer
        self.buffer = seq_buffer
        self.preprocess = preprocess
        self.postprocess = postprocess

    def __call__(self, human: Human, **kwargs):
        preproc_data = self.preprocess(human, skip_hip=True, **kwargs)
        transfo_data = self.transform(preproc_data)
        human = self.postprocess(transfo_data, source_human=human, **kwargs)
        reset_state = kwargs.get('reset', False)
        if reset_state:
            self.reset_state()
        return human

    def transform(self, human):
        human_seq, mask_seq = self.buffer(human)
        output_seq = self.transformer.predict_poses(human_seq, mask_seq)[0]
        # Return the last pose
        return output_seq[0, -1]

    def reset_state(self):
        self.buffer.reset_state()
