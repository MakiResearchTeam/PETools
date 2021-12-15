from typing import Dict, Mapping, Tuple, List
import numpy as np

from petools.core import PosePredictorInterface


class ProportionsLengthCalculator:

    def __init__(self, proportions: Mapping[str, int], main_line_indx: List[Tuple[int, int]] = [(4, 10), (5, 11)]):
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

        """
        self._proportions = proportions
        self._main_line_indx = main_line_indx
        self._prev_value = 0

    def __call__(self, preds: Mapping[str, List[Tuple[int, dict, dict, Tuple[str, float]]]]) -> Dict[str, Dict[str, float]]:
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
            points_2d = np.asarray(list(single_pred_data['human_2d'].values()), dtype=np.float32)
            main_length = self.calculate_main_length(points_2d)

            for name_prop, prop_value in self._proportions.items():
                # Calculate proportion for single data
                length_s = self.calculate_length_with_main_length(main_length, prop_value)
                result_dict[name_prop] = length_s
            h_id_2_length_data[human_id] = result_dict

        return h_id_2_length_data

    def calculate_length_with_main_length(self, main_length: float, proportion_value: float):
        return proportion_value * main_length

    def calculate_main_length(self, data: np.ndarray) -> float:
        lines_length = []
        for p1, p2 in self._main_line_indx:
            p1_data, p2_data = data[p1], data[p2]
            if p1_data[-1] < 1e-3 or p2_data[-1] < 1e-3:
                continue  # One of the points - are not visible
            length_s = self._calculate_length_line(p1_data[:-1], p2_data[:-1])  # Skip prob value
            lines_length.append(float(length_s))  # Get rid of numpy type

        if len(lines_length) == 0:
            assert self._prev_value is not None
            return self._prev_value

        # Calc avg
        line_avg = sum(lines_length) / len(lines_length)
        self._prev_value = line_avg
        return line_avg

    def _calculate_length_line(self, start_point: np.ndarray, end_point: np.ndarray):
        return np.sqrt(np.sum(np.square(end_point - start_point)))
