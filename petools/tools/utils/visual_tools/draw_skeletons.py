import numpy as np

from petools.core import PosePredictorInterface
from petools.tools.estimate_tools.constants import CONNECT_KP
from .visualize_tools import draw_skeleton


def draw_skeletons_on_image(
        image: np.ndarray, predictions: dict, color=(255, 0, 0), thick=3,
        draw_pose_name: bool = False, pose_name_position: tuple = (100, 100),
        draw_pose_conf: bool = False, pose_conf_position: tuple = (120, 120)):
    """
    Draw skeletons from `preidctions` on certain `image`
    With parameters such as color and thick of the line

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

    return draw_skeleton(
        image.copy(), humans, connect_indexes=CONNECT_KP, color=color, thickness=thick,
        draw_pose_name=draw_pose_name, pose_name_position=pose_name_position,
        draw_pose_conf=draw_pose_conf, pose_conf_position=pose_conf_position,
        pose_name_list=pose_name_list, pose_conf_class_list=pose_conf_class_list
    )

