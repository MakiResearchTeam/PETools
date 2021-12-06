import copy
from typing import List, Mapping, Tuple, Dict

import numpy as np

from petools.core import PosePredictorInterface


class AngleCalculator:

    def __init__(self, name_angle_to_calc: List[str]):
        """

        Parameters
        ----------
        name_angle_to_calc : list
            List contains name of angle which must be calculated

        """
        self._name_angle_to_calc = name_angle_to_calc
        self._name_angle_2_func = {
            "mmmm_angle": AngleCalculator.calculate_angle_1,
        }

    def __call__(
            self, preds: Mapping[str, List[Tuple[int, dict, dict, Tuple[str, float]]]],
            lengths: Mapping[str, Mapping[str, float]]) -> Mapping[str, List[Tuple[int, dict, dict, Tuple[str, float], Dict[str, float]]]]:
        """

        Parameters
        ----------
        preds : dict
            Prediction from PosePredictor as
            {
                PosePredictorInterface.HUMANS: [
                    (id_human, dict_2d, dict_3d, classification_info),
                    ...,
                    (id_human, dict_2d, dict_3d, classification_info),
                ],
                PosePredictorInterface.TIME: value
            }
            id_human - id of human, int
            dict_2d - dict of 2d prediction {"pn": [val_x, val_y, val_prob]}
            dict_3d - dict of 3d prediction {"pn": [val_x, val_y, val_z, val_prob]}
            classification_info - tuple from classificator ("name_pose": prob_value)
        lengths : dict
            Dict with data {human_id: {"name_proportion": length_proportion} }
            Where `length_proportion` - float value

        Return
        ------
        tuple
            Same dict as `preds`, but also with angle info as dict {"name_angle": value} for each human

        """
        new_dict = copy.deepcopy(preds)

        for i, pred, pred_new in enumerate(zip(preds[PosePredictorInterface.HUMANS], new_dict)):
            human_id = str(pred[0])
            data_2d = np.asarray(list(pred[1].values()), dtype=np.float32)[:, :-1] # Skip prob values
            data_3d = np.asarray(list(pred[2].values()), dtype=np.float32)[:, :-1] # Skip prob values
            name_pose = str(pred[-1][0])

            # Take length for this human
            length_human = lengths.get(human_id)
            if length_human is None:
                continue # Skip it???? Dont know

            result_angle_dict = {}
            for name_angle, func in self._name_angle_2_func:
                result_angle_dict[name_angle] = func(
                    data_2d=data_2d, data_3d=data_3d,
                    name_pose=name_pose, lengths=length_human
                )

            new_tuple = (
                pred_new[0], pred_new[1], pred_new[2], pred_new[3], # Keep original dict on safe!
                result_angle_dict
            )
            new_dict[PosePredictorInterface.HUMANS][i] = new_tuple

        return new_dict

    @staticmethod
    def calculate_angle_1(
            data_2d: np.ndarray, data_3d: np.ndarray,
            name_pose: str, lengths: Mapping[str, float], **kwargs) -> float:
        pass

