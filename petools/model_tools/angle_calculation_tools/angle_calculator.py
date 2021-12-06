import copy
from typing import List, Mapping, Tuple, Dict

from petools.core import PosePredictorInterface
from .angle_math import *


class AngleCalculator:
    right_shoulder_angle = 'right_shoulder_angle'
    left_shoulder_angle = 'left_shoulder_angle'
    right_hip_angle = 'right_hip_angle'
    left_hip_angle = 'left_hip_angle'
    right_shoulder_normal_angle = 'right_shoulder_normal_angle'
    left_shoulder_normal_angle = 'left_shoulder_normal_angle'
    right_hip_normal_angle = 'right_hip_normal_angle'
    left_hip_normal_angle = 'left_hip_normal_angle'
    right_elbow_angle = 'right_elbow_angle'
    left_elbow_angle = 'left_elbow_angle'
    right_knee_angle = 'right_knee_angle'
    left_knee_angle = 'left_knee_angle'

    def __init__(self, name_angle_to_calc: List[str] = None):
        """

        Parameters
        ----------
        name_angle_to_calc : list
            List contains name of angle which must be calculated

        """
        self._name_angle_to_calc = name_angle_to_calc

        self._name_angle_2_func = {
            AngleCalculator.right_shoulder_angle: right_shoulder_angle,
            AngleCalculator.left_shoulder_angle: left_shoulder_angle,
            AngleCalculator.right_hip_angle: right_hip_angle,
            AngleCalculator.left_hip_angle: left_hip_angle,
            AngleCalculator.right_shoulder_normal_angle: right_shoulder_normal_angle,
            AngleCalculator.left_shoulder_normal_angle: left_shoulder_normal_angle,
            AngleCalculator.right_hip_normal_angle: right_hip_normal_angle,
            AngleCalculator.left_hip_normal_angle: left_hip_normal_angle,
            AngleCalculator.right_elbow_angle: right_elbow_angle,
            AngleCalculator.left_elbow_angle: left_elbow_angle,
            AngleCalculator.right_knee_angle: right_knee_angle,
            AngleCalculator.left_knee_angle: left_knee_angle
        }

        if self._name_angle_to_calc is None:
            self._name_angle_to_calc = list(self._name_angle_2_func.keys())

    def __call__(
            self, preds: Mapping[str, List[Tuple[int, dict, dict, Tuple[str, float]]]],
            lengths: Mapping[str, Mapping[str, float]]) -> Mapping[
        str, List[Tuple[int, dict, dict, Tuple[str, float], Dict[str, float]]]]:
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
            Contains lengths of limbs for each human in `preds` dictionary:
            {human_id: {"limb_name": length} }, `length` is float.

        Return
        ------
        tuple
            Same dict as `preds`, but also with angle info as dict {"name_angle": value} for each human

        """
        new_dict = copy.deepcopy(preds)

        for i, (pred, pred_new) in enumerate(
                zip(preds[PosePredictorInterface.HUMANS], new_dict[PosePredictorInterface.HUMANS])
        ):
            human_id = str(pred[0])
            points2d = pred[1]
            points3d = pred[2]

            # Take length for this human
            limb_lengths = lengths.get(human_id)
            if limb_lengths is None:
                print(f'Found no limb length info for human with id={human_id}')
                continue

            result_angle_dict = {}
            for name_angle in self._name_angle_to_calc:
                func = self._name_angle_2_func.get(name_angle)
                if func is None:
                    print(f'Dont know angle with name={name_angle}. '
                          f'Allowed angle names: {list(self._name_angle_2_func.keys())}.'
                          f'\nSkipping.')
                    continue

                result_angle_dict[name_angle] = func(
                    points2d=points2d, points3d=points3d,
                    limb_lengths=limb_lengths
                )

            new_tuple = (
                *pred_new,  # Keep original dict safe!
                result_angle_dict
            )
            new_dict[PosePredictorInterface.HUMANS][i] = new_tuple

        return new_dict
