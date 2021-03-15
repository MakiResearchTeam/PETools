import numpy as np


def modify_humans(humans: list, thr=0.1) -> list:
    return [
        interp_points(single_human.to_list(), thr)
        for i, single_human in enumerate(humans)
    ]


def interp_points(keypoints: list, thr=0.1) -> list:
    """
    Calculate additional keypoints

    Parameters
    ----------
    keypoint : list
        Shape (N, 3)
    thr : float
        Threash hold of keypoint

    Return
    -------
    new_keypoints
        Shape (N+3, 3)

    """
    keypoints = np.asarray(keypoints).reshape(-1, 3)

    # keypoints - shape (N, 3)
    # 24
    if keypoints[12][-1] > thr and keypoints[13][-1] > thr:
        kp24 = (keypoints[12] + keypoints[13]) / 2.0
    else:
        kp24 = np.zeros(3).astype(np.float32, copy=False)

    # 25
    if keypoints[6][-1] > thr and keypoints[7][-1] > thr:
        kp25 = (keypoints[6] + keypoints[7]) / 2.0
    else:
        kp25 = np.zeros(3).astype(np.float32, copy=False)

    # 26
    kp26 = np.zeros(3).astype(np.float32, copy=False)

    if keypoints[5][-1] > thr and keypoints[4][-1] > thr and keypoints[3][-1] > thr and keypoints[2][-1] > thr:
        kp26 = (keypoints[5] + keypoints[4] + keypoints[3] + keypoints[2]) / 4.0
    elif keypoints[5][-1] > thr and keypoints[4][-1] > thr:
        kp26 = (keypoints[5] + keypoints[4]) / 2.0
    elif keypoints[3][-1] > thr and keypoints[2][-1] > thr:
        kp26 = (keypoints[3] + keypoints[2]) / 2.0


    return np.concatenate(
        [
            keypoints[0:1],
            np.expand_dims(kp25, axis=0),
            keypoints[1:2],
            np.expand_dims(kp26, axis=0),
            keypoints[6:],
            np.expand_dims(kp24, axis=0),
        ],
        axis=0
    ).astype(np.float32, copy=False).tolist()

