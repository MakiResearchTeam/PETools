import numpy as np
import cv2

from petools.core import PosePredictorInterface
from petools.tools import Human
from petools.tools.estimate_tools.constants import CONNECT_KP


def draw_humans(
        image, humans: list, connect_indexes: list, color=(255, 0, 0), thickness=2,
        draw_pose_name: bool = False, pose_name_position: tuple = (100, 100),
        draw_pose_conf: bool = False, pose_conf_position: tuple = (120, 120),
        pose_name_list: list = None,  pose_conf_class_list: list = None):
    """
    Draws all the `humans` on the given image.
    Parameters
    ----------
    image : np.ndarray
        The image.
    humans : list
        A list of humans that are represented by arrays of shape [n_points, 3] or by Human objects.
    connect_indexes : list
        Contains pairs or point indices that are connected to each other.
    color : tuple
        Color of the skeletons.
    thickness : int
        Thickness of the "skeleton bones".
    draw_pose_name
    pose_name_position
    draw_pose_conf
    pose_conf_position
    pose_name_list
    pose_conf_class_list

    Returns
    -------

    """
    if draw_pose_conf:
        assert len(pose_name_list) == len(humans)

    if draw_pose_conf:
        assert len(pose_conf_class_list) == len(humans)

    for indx in range(len(humans)):
        human = humans[indx]

        if isinstance(human, Human):
            data = np.array(human.to_list()).reshape(-1, 3)
        else:
            data = np.array(human).reshape(-1, 3)

        for j in range(len(connect_indexes)):
            single = connect_indexes[j]
            single_p1 = data[single[0]]
            single_p2 = data[single[1]]

            if single_p1[-1] > 1e-3 and single_p2[-1] > 1e-3:

                p_1 = (int(single_p1[0]), int(single_p1[1]))
                p_2 = (int(single_p2[0]), int(single_p2[1]))

                cv2.line(image, p_1, p_2, color=color, thickness=thickness)

        if draw_pose_name and pose_name_list is not None:
            cv2.putText(
                image, str(pose_name_list[indx]), pose_name_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6
            )

        if draw_pose_conf and pose_conf_class_list is not None:
            cv2.putText(
                image, str(pose_conf_class_list[indx]), pose_conf_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6
            )

    return image


def draw_skeletons_on_image(
        image: np.ndarray, predictions: dict, color=(255, 0, 0), thick=3,
        draw_pose_name: bool = False, pose_name_position: tuple = (100, 100),
        draw_pose_conf: bool = False, pose_conf_position: tuple = (120, 120)):
    """
    Draw skeletons from `predictions` on the given `image`.

    Parameters
    ----------
    image : np.ndarray
        The image on which detection was performed
    predictions : dict
        Prediction on `image` from this class and method `predict`
    color : tuple
        Color of the line,
        By default equal to (255, 0, 0) - i.e. red line
    thick : int
        Thick of the line, by default equal to 3, in most cases this value is enough
    draw_pose_name : bool
        If true, then on video also will be pose name per frame
    pose_name_position : tuple
        Position of the pose name (X, Y)
    draw_pose_conf : bool
        If true, then confidence of pose by classificator will be shown per frame
    pose_conf_position : tuple
        Position of the conf (X, Y)

    Returns
    -------
    np.ndarray
        Image with skeletons on it

    """
    predictions_humans = predictions[PosePredictorInterface.HUMANS]
    # 1 - index for 2d points
    humans = [list(single_h[1].values()) for single_h in predictions_humans]
    if draw_pose_name:
        pose_name_list = [single_h[3] for single_h in predictions_humans]
    else:
        pose_name_list = None

    if draw_pose_conf:
        pose_conf_class_list = [single_h[4] for single_h in predictions_humans]
    else:
        pose_conf_class_list = None

    return draw_humans(
        image.copy(), humans, connect_indexes=CONNECT_KP, color=color, thickness=thick,
        draw_pose_name=draw_pose_name, pose_name_position=pose_name_position,
        draw_pose_conf=draw_pose_conf, pose_conf_position=pose_conf_position,
        pose_name_list=pose_name_list, pose_conf_class_list=pose_conf_class_list
    )

