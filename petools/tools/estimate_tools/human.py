import numpy as np

from .constants import NUMBER_OF_KEYPOINTS


class Human:
    """
    Store keypoints of the single human

    """
    __slots__ = ('body_parts_3d', 'body_parts', 'score', 'id', 'count_kp', 'np')

    def __init__(self, count_kp=NUMBER_OF_KEYPOINTS):
        """
        Init class to store keypoints of a single human

        Parameters
        ----------
        count_kp : int
            Number of keypoint of full human. By default equal to 24

        """
        self.body_parts_3d = {}
        self.body_parts = {}
        self.score = 0.0
        self.id = -1
        self.count_kp = count_kp
        self.np = None

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def compile_np(self):
        self.np = self.to_np()

    def set_3d(self, array_3d):
        """
        Set z axis for every x,y keypoint

        Parameters
        ----------
        array_3d : list or np.ndarray
            Array of 3d keypoints with shape (N, 4)

        """
        if self.count_kp != len(array_3d):
            raise TypeError(f"Wrong size of array `z_data`. " +
                            f"\nExpected size: {self.count_kp}, but {len(array_3d)} was received."
            )
        self.body_parts_3d = dict()

        for part_idx in range(self.count_kp):
            self.body_parts_3d[part_idx] = [
                float(array_3d[part_idx][0]),
                float(array_3d[part_idx][1]),
                float(array_3d[part_idx][2]),
                float(array_3d[part_idx][-1])
            ]

    def to_list(self, th_hold=0.2) -> list:
        """
        Transform keypoints stored in this class to list
        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2
        Returns
        -------
        list
            List with lenght NK * 3, where NK - Number of Keypoints,
            Where each:
            0-th element is responsible for x axis coordinate
            1-th for y axis
            2-th for visibility of the points (or probability)
            If keypoint is not visible or below `th_hold`, this keypoint will be filled with zeros
        """
        list_data = []
        for i in range(self.count_kp):
            take_single = self.body_parts.get(i)
            if take_single is None or take_single[-1] < th_hold:
                list_data += [0.0, 0.0, 0.0]
            else:
                list_data += [
                    self.body_parts[i][0],
                    self.body_parts[i][1],
                    self.body_parts[i][-1],
                ]

        return list_data

    def to_list_from3d(self, th_hold=0.2) -> list:
        """
        Transform 3d keypoints stored in this class to list

        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2

        Returns
        -------
        list
            List with lenght NK * 4, where NK - Number of Keypoints,
            Where each:
            0-th element is responsible for x axis coordinate
            1-th for y axis
            2-th for z axis
            3-th for visibility of the points (or probability)
            If keypoint is not visible or below `th_hold`, this keypoint will be filled with zeros

        """
        if len(self.body_parts_3d) == 0:
            return np.zeros((self.count_kp * 4), dtype=np.float32).tolist()

        list_data = []
        for i in range(self.count_kp):
            take_single = self.body_parts_3d.get(i)
            if take_single is None or take_single[-1] < th_hold:
                list_data += [0.0, 0.0, 0.0, 0.0]
            else:
                list_data += [
                    self.body_parts[i][0],
                    self.body_parts[i][1],
                    self.body_parts[i][2],
                    self.body_parts[i][-1],
                ]

        return list_data

    def to_dict(self, th_hold=0.2, skip_not_visible=False, key_as_int=False, prepend_p=False) -> dict:
        """
        Transform keypoints stored in this class to dict

        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2
        skip_not_visible : bool
            If equal to True, then values with low probability (or invisible)
            Will be skipped from final dict
        prepend_p: bool
            Prepends letter 'p' to the points' ids. Needed in production.

        Returns
        -------
        dict
            Dict of the keypoints,
            { NumKeypoints:   [x_coord, y_coord, score],
              NumKeypoints_1: [x_coord, y_coord, score],
              ..........................................
            }
            Where NumKeypoints, NumKeypoints_1 ... are string values responsible for index of the keypoint,
            x_coord - coordinate of the keypoint on X axis
            y_coord - coordinate of the keypoint on Y axis
            score - confidence of the neural network
            If keypoint is not visible or below `th_hold`, this keypoint will be filled with zeros

        """
        dict_data = {}
        if key_as_int:
            key_tr = lambda x: int(x)
        else:
            key_tr = lambda x: str(x)

        if prepend_p:
            key_tr = lambda x: f'p{x}'

        for i in range(self.count_kp):
            take_single = self.body_parts.get(i)
            if take_single is not None and take_single[-1] >= th_hold:
                dict_data.update({
                    key_tr(i): [take_single[0], take_single[1], take_single[-1]]
                })
            elif not skip_not_visible:
                dict_data.update({
                    key_tr(i): [0.0, 0.0, 0.0]
                })

        return dict_data

    def to_dict_from3d(self, th_hold=0.2, skip_not_visible=False, key_as_int=False, prepend_p=False) -> dict:
        """
        Transform 3d keypoints stored in this class to dict

        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2
        skip_not_visible : bool
            If equal to True, then values with low probability (or invisible)
            Will be skipped from final dict
        key_as_int : bool
            If true, then in final dict, keys will be int values
            By default strings are used
        prepend_p: bool
            Prepends letter 'p' to the points' ids. Needed in production.

        Returns
        -------
        dict
            Dict of the keypoints,
            { NumKeypoints:   [x_coord, y_coord, z_coord, score],
              NumKeypoints_1: [x_coord, y_coord, z_coord, score],
              ..........................................
            }
            Where NumKeypoints, NumKeypoints_1 ... are string values responsible for index of the keypoint,
            x_coord - coordinate of the keypoint on X axis
            y_coord - coordinate of the keypoint on Y axis
            z_coord - coordinate of the keypoint on Z axis
            score - confidence of the neural network
            If keypoint is not visible or below `th_hold`, this keypoint will be filled with zeros
        """

        dict_data = {}
        if key_as_int:
            key_tr = lambda x: int(x)
        else:
            key_tr = lambda x: str(x)

        if prepend_p:
            key_tr = lambda x: f'p{x}'

        if len(self.body_parts_3d) == 0:
            return dict([(key_tr(i), [0.0, 0.0, 0.0, 0.0]) for i in range(self.count_kp)])

        for i in range(self.count_kp):
            take_single = self.body_parts_3d.get(i)
            if take_single is not None and take_single[-1] >= th_hold:
                dict_data.update({
                    key_tr(i): [take_single[0], take_single[1], take_single[2], take_single[-1]]
                })
            elif not skip_not_visible:
                dict_data.update({
                    key_tr(i): [0.0, 0.0, 0.0, 0.0]
                })

        return dict_data

    def to_np(self, th_hold=0.2):
        """
        Transform keypoints stored in this class to numpy array with shape (N, 3),
        Where N - number of points

        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2

        Returns
        -------
        np.ndarray
            Array of keypoints with shape (N, 3),
            Where N - number of points

        """
        if self.np is not None:
            return self.np

        list_points = self.to_list(th_hold=th_hold)
        # (N, 3)
        return np.array(list_points, dtype=np.float32).reshape(-1, 3)

    def to_np_from3d(self, th_hold=0.2):
        """
        Transform 3d keypoints stored in this class to numpy array with shape (N, 3),
        Where N - number of points

        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2

        Returns
        -------
        np.ndarray
            Array of keypoints with shape (N, 4),
            Where N - number of points
        """
        if self.np is not None:
            return self.np

        list_points = self.to_list_from3d(th_hold=th_hold)
        # (N, 4)
        return np.array(list_points, dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def from_array(human_array):
        """
        Take points from `human_array` and create Human class with this points

        Parameters
        ----------
        human_array : np.ndarray or list
            Array of input points
            NOTICE! Input array must be with shape (N, 3) (N - number of points)
            Human will handle N keypoints from this array

        Returns
        -------
        Human
            Created Human object with points in `human_np`

        """
        if len(human_array) == 0:
            return

        if len(human_array[0]) != 3:
            raise ValueError("Wrong input shape of human array. Expected array with shape (N, 3), but" +
                             f"shape (N, {len(human_array[0])}) were given."
            )

        human_class = Human(count_kp=len(human_array))
        human_id = 0
        sum_probs = 0.0
        human_class.np = np.array(human_array, dtype=np.float32)

        for part_idx in range(len(human_array)):
            human_class.body_parts[part_idx] = [
                float(human_array[part_idx][0]),
                float(human_array[part_idx][1]),
                float(human_array[part_idx][-1])
            ]
            sum_probs += float(human_array[part_idx][-1])

        human_class.score = sum_probs / len(human_array)
        return human_class

    @staticmethod
    def from_array_3d(human_array):
        """
        Take points from `human_array` and create Human class with this points

        Parameters
        ----------
        human_array : np.ndarray or list
            Array of input points
            NOTICE! Input array must be with shape (N, 4) (N - number of points)
            Human will handle N keypoints from this array

        Returns
        -------
        Human
            Created Human object with points in `human_np`

        """
        if len(human_array) == 0:
            return

        if len(human_array[0]) != 4:
            raise ValueError("Wrong input shape of human array. Expected array with shape (N, 4), but" +
                             f"shape (N, {len(human_array[0])}) were given."
            )

        human_class = Human(count_kp=len(human_array))
        human_id = 0
        sum_probs = 0.0
        human_class.np = np.array(human_array, dtype=np.float32)

        for part_idx in range(len(human_array)):
            human_class.body_parts_3d[part_idx] = [
                float(human_array[part_idx][0]),
                float(human_array[part_idx][1]),
                float(human_array[part_idx][2]),
                float(human_array[part_idx][-1])
            ]
            sum_probs += float(human_array[part_idx][-1])

        human_class.score = sum_probs / len(human_array)
        return human_class

    @staticmethod
    def from_dict(human_dict):
        """
        Take points from `human_dict` and create Human class with this points

        Parameters
        ----------
        human_dict : dict
            Dict of input points
            Example:
            {
                0: [22.0, 23.0, 1.0],
                1: [10, 20, 0.2],
                ....
            }

        Returns
        -------
        Human
            Created Human object with points in `human_dict`

        """
        human_class = Human()
        human_id = 0
        sum_probs = 0.0
        human_class.score = 0.0

        for part_idx, v_arr in human_dict.items():
            human_class.body_parts[part_idx] = [
                float(v_arr[0]),
                float(v_arr[1]),
                float(v_arr[-1])
            ]
            sum_probs += float(v_arr[-1])
        if len(human_dict) >= 1:
            human_class.score = sum_probs / len(human_dict)
        return human_class

    @staticmethod
    def from_dict_3d(human_dict):
        """
        Take points from `human_dict` and create Human class with this points

        Parameters
        ----------
        human_dict : dict
            Dict of input points
            Example:
            {
                0: [22.0, 23.0, 1.0],
                1: [10, 20, 0.2],
                ....
            }

        Returns
        -------
        Human
            Created Human object with points in `human_dict`
        """
        human_class = Human()
        human_id = 0
        sum_probs = 0.0
        human_class.score = 0.0

        for part_idx, v_arr in human_dict.items():
            human_class.body_parts_3d[part_idx] = [
                float(v_arr[0]),
                float(v_arr[1]),
                float(v_arr[2]),
                float(v_arr[-1])
            ]
            sum_probs += float(v_arr[-1])
        if len(human_dict) >= 1:
            human_class.score = sum_probs / len(human_dict)
        return human_class

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()

