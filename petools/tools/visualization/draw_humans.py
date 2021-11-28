import numpy as np
import cv2
from random import randint

from petools.core import PosePredictorInterface
from .core import draw_human
from petools.tools.estimate_tools.constants import CONNECT_KP


def random_color():
    return randint(0, 255), randint(0, 255), randint(0, 255)


ID2COLOR = {
    '0': (255, 0, 0),
    '1': (0, 255, 0),
    '2': (0, 0, 255),
    '3': (0, 0, 0),
    '4': (255, 255, 0),
    '5': (255, 255, 255),
    '6': (20, 100, 255),
    '7': (220, 100, 255),
    '8': (20, 200, 255),
    '9': (220, 100, 255),
}


def draw_humans(
        image, humans, humans_ids: list = None,
        connect_indexes: list = CONNECT_KP,
        color=(255, 0, 0), thickness=2,
        conf_threshold: float = 1e-3,
        draw_pose_name: bool = False, pose_name_position: tuple = (100, 100),
        draw_pose_conf: bool = False, pose_conf_position: tuple = (100, 170),
        pose_name_list: list = None, pose_conf_list: list = None,
        do_coherence_check: bool = True,
        root_points: tuple = (2, 0, 1, 4, 5, 10, 11, 22)
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
    humans_ids : list, optional
        Contains IDs for the corresponding humans.
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
    do_coherence_check : bool
        Whether to perform human skeleton coherence check.
    root_points : tuple
        A list of root points used in the coherence check. From those points tracing will be performed
        to mask the absent (unconnected) part of the skeleton graph.

    Warnings
    --------
    Currently the method draws all the pose info (name and confidence) in one place.
    Please be aware as in case of drawing multiple humans the info for each one will overlap.
    """
    if isinstance(humans, dict):
        humans_ = humans.get(PosePredictorInterface.HUMANS)
        assert humans_ is not None, f"Received dictionary which does not contain '{PosePredictorInterface.HUMANS}' key. " \
                        f'Received dict={humans}'
        humans_ids = [id for id, human_2d, human_3d, pose_info in humans_]
        humans = [human_2d for id, human_2d, human_3d, pose_info in humans_]

        if draw_pose_name:
            pose_name_list = [pose_info[0] for id, human_2d, human_3d, pose_info in humans_]
        if draw_pose_conf:
            pose_conf_list = [pose_info[1] for id, human_2d, human_3d, pose_info in humans_]

    # No humans to draw
    if len(humans) == 0:
        return

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

        if humans_ids:
            id = str(humans_ids[i])
            color = ID2COLOR.get(id)
            if not color:
                color = random_color()
                ID2COLOR[id] = color

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
            pose_conf_position=pose_conf_position,
            do_coherence_check=do_coherence_check,
            root_points=root_points
        )
