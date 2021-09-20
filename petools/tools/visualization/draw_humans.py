import numpy as np
import cv2

from petools.core import PosePredictorInterface
from .core import draw_human
from petools.tools.estimate_tools.constants import CONNECT_KP


def draw_humans(
        image, humans, connect_indexes: list = CONNECT_KP,
        color=(255, 0, 0), thickness=2,
        conf_threshold: float = 1e-3,
        draw_pose_name: bool = False, pose_name_position: tuple = (100, 100),
        draw_pose_conf: bool = False, pose_conf_position: tuple = (100, 170),
        pose_name_list: list = None, pose_conf_list: list = None,
):
    """
    Draws all the `humans` on the given image inplace. If you don't want the original image to be modified,
    pass in a copy of it.

    Parameters
    ----------
    image : np.ndarray
        The image.
    humans : list or dict
        If it's a list, it must contain all the humans (Human, list, dict) to draw.
        If it's a dict, it must be a dictionary returned by the PosePredictor instance.
    connect_indexes : list
        Contains pairs or point indices that are connected to each other.
    color : tuple
        Color of the skeletons.
    thickness : int
        Thickness of the "skeleton bones".
    conf_threshold : float
        Minimum threshold a pair of points must pass to be drawn.
    draw_pose_name : bool
        Determines whether to draw the pose name.
    pose_name_position : tuple
        Position where to draw the pose name.
    draw_pose_conf : bool
        Determines whether to draw the pose confidence.
    pose_conf_position : tuple
        Position where to draw the pose name.
    pose_name_list : list, optional
        List of (class) names for the humans' poses.
    pose_conf_list : list
        List of confidence values for each pose (class) name.

    Warnings
    --------
    Currently the method draws all the pose info in one place. Please be aware as in case of drawing multiple
    humans the info for each one will overlap.
    """
    if isinstance(humans, dict):
        humans_ = humans.get(PosePredictorInterface.HUMANS)
        assert humans_, f'Received dictionary which does not contain {PosePredictorInterface.HUMANS} key. ' \
                        f'Received dict={humans_}'
        humans = [human_2d for id, human_2d, human_3d, pose_info in humans_]
        pose_name_list = [pose_info[0] for id, human_2d, human_3d, pose_info in humans_]
        pose_conf_list = [pose_info[1] for id, human_2d, human_3d, pose_info in humans_]

    if pose_name_list:
        assert len(pose_name_list) == len(humans)

    if pose_conf_list:
        assert len(pose_conf_list) == len(humans)

    for i, human in enumerate(humans):
        pose_name = None
        if pose_name_list and draw_pose_name:
            pose_name = pose_name_list[i]

        pose_conf = None
        if pose_conf_list and draw_pose_name:
            pose_conf = pose_conf_list[i]

        draw_human(
            image=image,
            human=human,
            connect_indices=connect_indexes,
            color=color,
            thickness=thickness,
            conf_threshold=conf_threshold,
            pose_name=pose_name,
            pose_name_position=pose_name_position,
            pose_conf=pose_conf,
            pose_conf_position=pose_conf_position
        )
