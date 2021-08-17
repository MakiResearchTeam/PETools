import os
import json
import numpy as np

from petools.tools import Human


class Postprocess2DPose:

    NONE = 'unknown_class'

    def __init__(self, path_to_config: str):
        """

        Parameters
        ----------
        path_to_config : str
            Path to config data (json file) for postprocessing

        """
        assert os.path.isfile(path_to_config), f"Could not find config file in {path_to_config}."
        with open(path_to_config, 'r') as fp:
            class2name = json.load(fp)

        self._class2name = class2name

    def __call__(self, nn_predict: np.ndarray, source_human: Human, **kwargs) -> Human:
        print(nn_predict.shape)
        indx_max_conf = int(np.argmax(nn_predict, axis=1))
        # Poses in config file shifted by 1
        shifted_indx = indx_max_conf + 1

        if self._class2name.get(str(shifted_indx)) is None:
            source_human.pose_name = self.NONE
        else:
            source_human.pose_name = self._class2name[str(shifted_indx)]
        source_human.pose_class_conf = nn_predict[0, indx_max_conf]
        return source_human
