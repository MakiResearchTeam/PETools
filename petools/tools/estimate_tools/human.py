import numpy as np

from .constants import NUMBER_OF_KEYPOINTS


class Human:
    """
    Store keypoints of the single human

    """
    __slots__ = ('body_parts', 'score', 'id', 'count_kp', 'np', 'np3d')

    def __init__(self, count_kp=NUMBER_OF_KEYPOINTS):
        """
        Init class to store keypoints of a single human

        Parameters
        ----------
        count_kp : int
            Number of keypoint of full human. By default equal to 24

        """
        self.body_parts = {}
        self.score = 0.0
        self.id = -1
        self.count_kp = count_kp
        self.np = self.np3d = None

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def compile_np(self):
        self.np = self.to_np()

    def compile_np_3d(self):
        self.np3d = self.to_np_from3d()

    def set_3d(self, array_3d):
        """
        Set z axis for every x,y keypoint

        Parameters
        ----------
        array_3d : list or np.ndarray
            Array of 3d keypoints with shape (N, 4)

        """
        if self.count_kp != len(array_3d):
            raise TypeError(
                f"Wrong size of array `array_3d`. " +
                f"\nExpected size: {self.count_kp}, but {len(array_3d)} was received."
            )
        self.np3d = array_3d.astype(np.float32, copy=False)

    def to_list(self, th_hold=0.2) -> list:
        """
        Transform keypoints stored in this class to list
        NOTICE! This method ussaly used in order to init values of Human class
        After getting prediction from pafprocess module
        In order to take array of keypoins, use `to_np`

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

    def to_list_from3d(self, th_hold=0.2, convert_to_list=False) -> list:
        """
        Transform 3d keypoints stored in this class to list

        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2
        convert_to_list : bool
            Final array will be created from numpy array, at the end of function array is converted to list.
            If false, then conversion will be not applied, otherwise it will be converted to list.

        Returns
        -------
        np.ndarray
            If `convert_to_list` is true,
            Shape - (N, 4)
            where:
                N - number of keypoints
                0 indx of final axis - x axis
                1 indx of final axis - y axis
                2 indx of final axis - z axis
                3 indx of final axis - v (visibility) axis

        list
            List with lenght NK * 4, where NK - Number of Keypoints,
            Where each:
            0-th element is responsible for x axis coordinate
            1-th for y axis
            2-th for z axis
            3-th for visibility of the points (or probability)
            If keypoint is not visible or below `th_hold`, this keypoint will be filled with zeros

        """
        if self.np3d is None:
            raise ValueError(
                "Error! np3d array in Human is not compiled. Call `compile_np_3d` in order to compile 3d array. \n"
                "Compile it before call method `to_list_from3d`."
            )
        # In order to avoid interactions on np3d from outside
        new_arr_3d = self.np3d.copy()
        # If some points below th_hold - this keypoints are not visible
        # Make them equal to 0.0
        kp_3d_good = self.np3d[:, -1] < th_hold
        new_arr_3d[kp_3d_good] = 0.0
        # Return np array if it needed
        if not convert_to_list:
            return new_arr_3d
        # Flatten array into vector
        new_arr_3d = new_arr_3d.reshape(-1)
        # Return list
        return new_arr_3d.tolist()

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
        elif prepend_p:
            key_tr = lambda x: f'p{x}'
        else:
            key_tr = lambda x: str(x)

        if self.np is None:
            raise ValueError(
                "Error! np3в array in Human is not compiled. Call `compile_np` in order to compile 3d array. \n"
                "Compile it before call method `to_dict`."
            )
        # If array was cached, then use it in order to create dict representation
        # Thats because cached array can be changed while body_parts - not
        for i in range(self.count_kp):
            take_single = self.np[i]
            if take_single[-1] >= th_hold:
                dict_data.update({
                    key_tr(i): [float(take_single[0]), float(take_single[1]), float(take_single[-1])]
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
        elif prepend_p:
            key_tr = lambda x: f'p{x}'
        else:
            key_tr = lambda x: str(x)

        if self.np3d is None:
            raise ValueError(
                "Error! np3d array in Human is not compiled. Call `compile_np_3d` in order to compile 3d array. \n"
                "Compile it before call method `to_dict_from3d`."
            )

        # If array was cached, then use it in order to create dict representation
        # Thats because cached array can be changed while body_parts - not
        for i in range(self.count_kp):
            take_single = self.np3d[i]
            if take_single[-1] >= th_hold:
                dict_data.update({
                    key_tr(i): [
                        float(take_single[0]), float(take_single[1]),
                        float(take_single[2]), float(take_single[-1])
                    ]
                })
            elif not skip_not_visible:
                dict_data.update({
                    key_tr(i): [0.0, 0.0, 0.0, 0.0]
                })

        return dict_data

    def to_np(self, th_hold=0.2, copy_if_cached=False):
        """
        Transform keypoints stored in this class to numpy array with shape (N, 3),
        Where N - number of points

        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2
        copy_if_cached : bool
            If True, then if array of keypoints are cached, it will be copied,
            In order to safe original (saved in this class) array

        Returns
        -------
        np.ndarray
            Array of keypoints with shape (N, 3),
            Where N - number of points

        """
        if self.np is not None:
            if copy_if_cached:
                return self.np.copy()
            return self.np

        list_points = self.to_list(th_hold=th_hold)
        # (N, 3)
        return np.array(list_points, dtype=np.float32).reshape(-1, 3)

    def to_np_from3d(self, th_hold=0.2, copy_if_cached=False):
        """
        Transform 3d keypoints stored in this class to numpy array with shape (N, 3),
        Where N - number of points

        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2
        copy_if_cached : bool
            If True, then if array of keypoints are cached, it will be copied,
            In order to safe original (saved in this class) array

        Returns
        -------
        np.ndarray
            Array of keypoints with shape (N, 4),
            Where N - number of points
        """
        if self.np3d is not None:
            if copy_if_cached:
                return self.np3d.copy()
            return self.np3d

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
        human_class.np = np.asarray(human_array, dtype=np.float32)
        human_class.score = float(np.sum(human_array[:, -1], axis=0)) / len(human_array)
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
        human_class.np3d = np.asarray(human_array, dtype=np.float32)
        human_class.score = float(np.sum(human_array[:, -1], axis=0)) / len(human_array)
        return human_class

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()

