import numpy as np
from petools.tools.estimate_tools.human import Human


def modify_humans(humans: list, thr=0.1) -> list:
    return [
        interp_points(single_human, thr)
        for i, single_human in enumerate(humans)
    ]


def interp_points(keypoints_h: Human, thr=0.1) -> np.ndarray:
    """
    Calculate additional keypoints_h.np

    Parameters
    ----------
    keypoints_h : np.ndarray
        Shape (N, 3)
    thr : float
        Threash hold of keypoint

    Return
    -------
    new_keypoints_h
        Shape (N+3, 3)

    """
    # 24
    if keypoints_h.np[12, -1] > thr and keypoints_h.np[13, -1] > thr:
        kp24 = (keypoints_h.np[12] + keypoints_h.np[13]) / 2.0
    else:
        kp24 = np.zeros(3, dtype=np.float32)

    # 25.
    if keypoints_h.np[6, -1] > thr and keypoints_h.np[7, -1] > thr:
        kp25 = (keypoints_h.np[6] + keypoints_h.np[7]) / 2.0
    else:
        kp25 = np.zeros(3, dtype=np.float32)

    # 26
    kp26 = np.zeros(3, dtype=np.float32)
    if keypoints_h.np[5, -1] > thr and keypoints_h.np[4, -1] > thr and keypoints_h.np[3, -1] > thr and keypoints_h.np[2, -1] > thr:
        kp26 = (keypoints_h.np[5] + keypoints_h.np[4] + keypoints_h.np[3] + keypoints_h.np[2]) / 4.0
    elif keypoints_h.np[5, -1] > thr and keypoints_h.np[4, -1] > thr:
        kp26 = (keypoints_h.np[5] + keypoints_h.np[4]) / 2.0
    elif keypoints_h.np[3, -1] > thr and keypoints_h.np[2, -1] > thr:
        kp26 = (keypoints_h.np[3] + keypoints_h.np[2]) / 2.0

    return np.concatenate(
        [
            keypoints_h.np[0:1],
            np.expand_dims(kp25, axis=0),
            keypoints_h.np[1:2],
            np.expand_dims(kp26, axis=0),
            keypoints_h.np[6:],
            np.expand_dims(kp24, axis=0),
        ],
        axis=0
    )

