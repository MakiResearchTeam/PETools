import numpy as np

from .classifier import Classifier
from petools.tools import Human
from .process_data import Preprocess2DPose, Postprocess2DPose


class PoseClassifier:
    def __init__(
            self,
            classifier: Classifier,
            preprocess: Preprocess2DPose,
            postprocess: Postprocess2DPose
    ):
        """
        Generic class for instance transforming a pose into another pose (possibly with another dimensionality).

        Parameters
        ----------
        classifier : Classifier
            pass
        preprocess : Preprocess2DPose
            pass
        postprocess : Postprocess2DPose
            pass

        """
        self.classifier = classifier
        self.preprocess = preprocess
        self.postprocess = postprocess

    def __call__(self, human: Human, **kwargs):
        preproc_data = self.preprocess(human, **kwargs)
        transfo_data = self.transform(preproc_data)
        out_human = self.postprocess(transfo_data, source_human=human, **kwargs)
        return out_human

    def transform(self, human: np.ndarray):
        batched_h = np.expand_dims(human, axis=0)
        classified_data = np.asarray(self.classifier.classify_poses(batched_h), dtype=np.float32)
        return classified_data
