import copy
from typing import Dict, Mapping, Tuple, List
import numpy as np

from petools.core import PosePredictorInterface
from .proportions_length_calculator import ProportionsLengthCalculator
from .constants import PROPORTIONS_INDX


class DynamicLengthCalculator:
    MAX_NEUTRAL_POSE_COUNT = 30
    MAX_GRAB_STATS_COUNT = 90

    def __init__(
            self, proportions: Mapping[str, int], main_line_indx: List[Tuple[int, int]] = [(4, 10), (5, 11)],
            name_neutral_pose: str = 'netralnya_poza', pose_conf_threshold: float = 0.95):
        """

        Parameters
        ----------
        proportions : dict
            Store data as {"name_proportion": value }
            Where:
                `value` - proportion value, float/int value
        main_line_indx : list
            Points which index will be as length of the MAIN-LINE.
            Length of the MAIN-LINE mean what all other lines length in proportion will be calculated with this line.
            `main_line_indx` store list of tuples, where each tuple its - (indx_1, indx_2), i.e. its indx in order to
            take line from skeleton, if there is more than 1 tuple, final length of the MAIN-LINE will be as AVG of there lines.
            These lines will be calculated each call.
            By default taken next points:
                4 - Left shoulder
                10 - Left hip
                5 - Right shoulder
                11 - Right hip
        pose_conf_threshold : float
            A neutral pose is considered a valid neutral pose if its confidence exceeds the threshold.
        """
        self._proportion_length_calculator = ProportionsLengthCalculator(
            proportions=proportions, main_line_indx=main_line_indx
        )
        self._name_neutral_pose = name_neutral_pose
        self._pose_conf_threshold = pose_conf_threshold
        # For every dict below - {"id of human" : value}
        self._counter_neutral_pose_dict: Dict[str, int] = dict()
        self._temp_collected_stats_dict: Dict[str, dict] = dict()
        self._counter_grab_stats_dict: Dict[str, int] = dict()
        self._stats_are_ready_dict: Dict[str, bool] = dict()
        self._stats_for_human_dict: Dict[str, dict] = dict()

    def __call__(self, preds: Mapping[str, List[Tuple[int, dict, dict, Tuple[str, float]]]]) -> Dict[
        str, Dict[str, float]]:
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

        Return
        ------
        dict
            Return dict with data {human_id: {"name_proportion": length_proportion} }
            Where `length_proportion` - float value

        """
        h_id_2_length_data = {}
        for single_pred_data in preds[PosePredictorInterface.HUMANS]:
            result_dict = {}
            human_id = str(single_pred_data['human_id'])
            # Check pose
            pose_name, pose_conf = single_pred_data['human_pose_name'], single_pred_data['human_pose_confidence']
            if pose_name == self._name_neutral_pose and pose_conf >= self._pose_conf_threshold:
                # +1 for counter if pose is neutral
                is_human_here = self._counter_neutral_pose_dict.get(human_id)
                if is_human_here is None:
                    self._counter_neutral_pose_dict[human_id] = 0
                self._counter_neutral_pose_dict[human_id] = min(
                    self._counter_neutral_pose_dict[human_id] + 1,
                    self.MAX_NEUTRAL_POSE_COUNT
                )

                # Check if neutral poses are more than param
                if self._counter_neutral_pose_dict[human_id] == self.MAX_NEUTRAL_POSE_COUNT:
                    # We should start grab stats for this person or continue to grab
                    is_human_stats_here = self._temp_collected_stats_dict.get(human_id)
                    if is_human_stats_here is None:
                        self._temp_collected_stats_dict[human_id] = dict()  # dict for each limb with length info
                        self._counter_grab_stats_dict[human_id] = 0
                    self._counter_grab_stats_dict[human_id] = min(
                        self._counter_grab_stats_dict[human_id] + 1,
                        self.MAX_GRAB_STATS_COUNT
                    )

                    # Here must be some collector stuff from preds
                    length_at_current_state = self.calculate_length_from_pred(single_pred_data)
                    self._temp_collected_stats_dict[human_id] = self.append_dict_info_another(
                        dict_1=self._temp_collected_stats_dict[human_id],
                        dict_2=length_at_current_state
                    )
                    if self._counter_grab_stats_dict[human_id] == self.MAX_GRAB_STATS_COUNT:
                        # We must avg collected data and assign to dict
                        self._stats_for_human_dict[human_id] = self.calculate_avg_length(
                            dict_stats=self._temp_collected_stats_dict[human_id]
                        )
                        self._stats_are_ready_dict[human_id] = True
                        # Drop counters
                        self._counter_grab_stats_dict[human_id] = 0
                        self._counter_neutral_pose_dict[human_id] = 0
                        self._temp_collected_stats_dict[human_id] = dict()
            else:
                # Another pose
                # We must drop all stats for this person, if its here
                # But keep stats which already calculated for person (for future use)
                if self._counter_neutral_pose_dict.get(human_id) is not None:
                    del self._counter_neutral_pose_dict[human_id]
                if self._temp_collected_stats_dict.get(human_id) is not None:
                    del self._temp_collected_stats_dict[human_id]
                if self._counter_grab_stats_dict.get(human_id) is not None:
                    del self._counter_grab_stats_dict[human_id]

            # Check stats
            is_human_stats_ready = self._stats_are_ready_dict.get(human_id)
            if is_human_stats_ready is None:
                h_id_2_length_data[human_id] = self._proportion_length_calculator(
                    preds={
                        PosePredictorInterface.HUMANS: [single_pred_data]
                    }
                )[human_id]
            else:
                h_id_2_length_data[human_id] = copy.deepcopy(self._stats_for_human_dict[human_id])

        return h_id_2_length_data

    def calculate_avg_length(self, dict_stats: Mapping[str, list]) -> Dict[str, float]:
        avg_length_dict = dict()
        for k, v_list in dict_stats.items():
            v_np = np.asarray(v_list, dtype=np.float32)
            std, mean = v_np.std(), v_np.mean()
            selected = np.bitwise_and(
                mean - 3 * std <= v_np,
                v_np <= mean + 3 * std
            )
            good_lengths = v_np[selected]
            avg_length = good_lengths.mean()
            avg_length_dict[k] = avg_length

        return avg_length_dict

    def calculate_length_from_pred(self, pred: dict) -> Dict[str, float]:
        name_limb2length = dict()
        points_2d = np.asarray(list(pred['human_2d'].values()), dtype=np.float32)
        for name_prop, indxes_list in PROPORTIONS_INDX.items():
            all_lengths = []
            for p1_i, p2_i in indxes_list:
                p1 = points_2d[p1_i].copy()
                p2 = points_2d[p2_i].copy()
                if p1[-1] < 1e-3 or p2[-1] < 1e-3:
                    continue  # One of the points - are not visible
                length = self._calculate_length_line(p1, p2)
                all_lengths.append(length)
            if len(all_lengths) != 0:
                avg_length = sum(all_lengths) / len(all_lengths)
            else:
                avg_length = None
            name_limb2length[name_prop] = avg_length

        return name_limb2length

    def append_dict_info_another(self, dict_1: Mapping[str, list], dict_2: Mapping[str, float]) -> Dict[str, list]:
        if len(dict_1) == 0:
            # if original is empty, add dict 2 into it and return
            for k, v in dict_2.items():
                if v is None:
                    dict_1[k] = []
                else:
                    dict_1[k] = [v]
            return dict_1
        # Append
        for k, v in dict_1.items():
            if dict_2[k] is not None:
                dict_1[k] += [dict_2[k]]

        return dict_1

    def _calculate_length_line(self, start_point: np.ndarray, end_point: np.ndarray):
        return np.sqrt(np.sum(np.square(end_point - start_point), axis=-1))
