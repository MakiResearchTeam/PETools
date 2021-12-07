import math
import numpy as np


def compute_cuboid_diag(p1: list, p2: list, diag3d_length: float, p1_closer: bool):
    """
    Finds a 3D diagonal of a cuboid which 2D diagonal is defined by vector (p1, p2).

    Parameters
    ----------
    p1 : list
        Center of coordinate system of the cuboid.
    p2 : list
        A corner of the cuboid that stretches from the p1 point.
    diag3d_length : float
        Length of the 3D diagonal of the cuboid.

    Returns
    -------
    list
        A 3D vector (x, y, z) representing diagonal of the cuboid.
    """
    # 3D vector pointing from ls to rs fully defines a cuboid and is represented by the
    # diagonal of that cuboid.
    # This function uses this fact in order to compute the cuboids diagonal

    x = p2[0] - p1[0]
    y = p2[1] - p1[1]

    dist_squared = x ** 2 + y ** 2
    diff = diag3d_length ** 2 - dist_squared
    z = math.sqrt(abs(diff))
    # Z axis is going away from camera (closer to the camera, smaller the z)
    if not p1_closer:
        z *= -1
    return [x, y, z]


def compute_angle_2vec(p1: list, p2: list, p3: list, p1p2_dist: float, p2p3_dist: float):
    """
    Computes angle between vectors p2-p1 and p2-p3.

    Parameters
    ----------
    p1 : list
        [x, y, z]
    p2 : list
        [x, y, z]
    p3 : list
        [x, y, z]
    p1p2_dist : float
        Distance between points p1 and p2 in 3D space.
    p2p3_dist : float
        Distance between points p2 and p3 in 3D space.

    Returns
    -------
    float
        Angle between vectors p2-p1 and p2-p3.
    """
    # v1 usually corresponds to a vector lying on shoulders or on pelvis.
    v1 = compute_cuboid_diag(p2, p1, p1p2_dist, p2[2] < p1[2])
    # v2 usually corresponds to a vector pointing to an elbow or to a knee.
    v2 = compute_cuboid_diag(p2, p3, p2p3_dist, False)
    prod = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    return float(np.arccos(prod) * 180 / np.pi)


# for knees and elbows
def compute_angle_2vec_V2(p1: list, p2: list, p3: list, p1p2_dist: float, p2p3_dist: float):
    """
    Computes angle between vectors p2-p1 and p2-p3.

    Parameters
    ----------
    p1 : list
        [x, y, z]
    p2 : list
        [x, y, z]
    p3 : list
        [x, y, z]
    p1p2_dist : float
        Distance between points p1 and p2 in 3D space.
    p2p3_dist : float
        Distance between points p2 and p3 in 3D space.

    Returns
    -------
    float
        Angle between vectors p2-p1 and p2-p3.
    """
    # v1 usually corresponds to a vector lying on shoulders or on pelvis.
    v1 = compute_cuboid_diag(p2, p1, p1p2_dist, True)
    # v2 usually corresponds to a vector pointing to an elbow or to a knee.
    v2 = compute_cuboid_diag(p2, p3, p2p3_dist, False)
    prod = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    return float(np.arccos(prod) * 180 / np.pi)


def compute_angle_1vec(p1, p2, p1p2_dist):
    """
    Computes angle between vectors p1-p2 and [0, 1, 0] (normal vector pointing down).

    Parameters
    ----------
    p1 : list
        [x, y, z]
    p2 : list
        [x, y, z]
    p1p2_dist : float
        Distance between points p1 and p2 in 3D space.

    Returns
    -------
    float
        Angle between vectors p1-p2 and p2-p3 and [0, 1, 0] in 3D space.
    """
    # v1 usually corresponds to a vector pointing to an elbow or a knee.
    v1 = compute_cuboid_diag(p1, p2, p1p2_dist, False)
    prod = np.dot([0, 1, 0], v1) / np.linalg.norm(v1)
    return float(np.arccos(prod) * 180 / np.pi)


def extract_point(points2d, points3d, key):
    """
    Extracts x, y, z coordinates for a given `key`.
    """
    p2d = points2d[key]
    p3d = points3d[key]
    return [p2d[0], p2d[1], p3d[2]]


def angle2vecs(points2d, points3d, points_keys, limb_lengths, limb_lengths_keys):
    p1 = extract_point(points2d, points3d, points_keys[0])
    p2 = extract_point(points2d, points3d, points_keys[1])
    p3 = extract_point(points2d, points3d, points_keys[2])
    p2p1_dist = limb_lengths[limb_lengths_keys[0]]
    p2p3_dist = limb_lengths[limb_lengths_keys[1]]
    return compute_angle_2vec(p1, p2, p3, p2p1_dist, p2p3_dist)


# for knees and elbows
def angle2vecs_V2(points2d, points3d, points_keys, limb_lengths, limb_lengths_keys):
    p1 = extract_point(points2d, points3d, points_keys[0])
    p2 = extract_point(points2d, points3d, points_keys[1])
    p3 = extract_point(points2d, points3d, points_keys[2])
    p2p1_dist = limb_lengths[limb_lengths_keys[0]]
    p2p3_dist = limb_lengths[limb_lengths_keys[1]]
    return compute_angle_2vec_V2(p1, p2, p3, p2p1_dist, p2p3_dist)


def angle1vec(points2d, points3d, points_keys, limb_lengths, limb_lengths_key):
    p1 = extract_point(points2d, points3d, points_keys[0])
    p2 = extract_point(points2d, points3d, points_keys[1])
    p2p1_dist = limb_lengths[limb_lengths_key]
    return compute_angle_1vec(p1, p2, p2p1_dist)


def right_shoulder_angle(points2d, points3d, limb_lengths):
    # Left shoulder, right shoulder, right elbow
    point_keys = ['p4', 'p5', 'p7']
    lengths_keys = ['ss', 'se']
    return angle2vecs(points2d, points3d, point_keys, limb_lengths, lengths_keys)


def left_shoulder_angle(points2d, points3d, limb_lengths):
    # Right shoulder, left shoulder, left elbow
    point_keys = ['p5', 'p4', 'p6']
    lengths_keys = ['ss', 'se']
    return angle2vecs(points2d, points3d, point_keys, limb_lengths, lengths_keys)


def right_hip_angle(points2d, points3d, limb_lengths):
    # Left hip, right hip, right knee
    point_keys = ['p10', 'p11', 'p13']
    lengths_keys = ['hh', 'hk']
    return angle2vecs(points2d, points3d, point_keys, limb_lengths, lengths_keys)


def left_hip_angle(points2d, points3d, limb_lengths):
    # Right hip, left hip, left knee
    point_keys = ['p11', 'p10', 'p12']
    lengths_keys = ['hh', 'hk']
    return angle2vecs(points2d, points3d, point_keys, limb_lengths, lengths_keys)


def right_shoulder_normal_angle(points2d, points3d, limb_lengths):
    # Right shoulder, right elbow
    point_keys = ['p5', 'p7']
    lengths_key = 'se'
    return angle1vec(points2d, points3d, point_keys, limb_lengths, lengths_key)


def left_shoulder_normal_angle(points2d, points3d, limb_lengths):
    # Left shoulder, left elbow
    point_keys = ['p4', 'p6']
    lengths_key = 'se'
    return angle1vec(points2d, points3d, point_keys, limb_lengths, lengths_key)


def right_hip_normal_angle(points2d, points3d, limb_lengths):
    # Right hip, right knee
    point_keys = ['p11', 'p13']
    lengths_key = 'hk'
    return angle1vec(points2d, points3d, point_keys, limb_lengths, lengths_key)


def left_hip_normal_angle(points2d, points3d, limb_lengths):
    # Left hip, left knee
    point_keys = ['p10', 'p12']
    lengths_key = 'hk'
    return angle1vec(points2d, points3d, point_keys, limb_lengths, lengths_key)


def right_elbow_angle(points2d, points3d, limb_lengths):
    # Right shoulder, right elbow, right wrist
    point_keys = ['p5', 'p7', 'p9']
    lengths_keys = ['se', 'ew']
    return angle2vecs_V2(points2d, points3d, point_keys, limb_lengths, lengths_keys)


def left_elbow_angle(points2d, points3d, limb_lengths):
    # Left shoulder, left elbow, left wrist
    point_keys = ['p4', 'p6', 'p8']
    lengths_keys = ['se', 'ew']
    return angle2vecs_V2(points2d, points3d, point_keys, limb_lengths, lengths_keys)


def right_knee_angle(points2d, points3d, limb_lengths):
    # Right hip, right knee, right ankle
    point_keys = ['p11', 'p13', 'p15']
    lengths_keys = ['hk', 'ka']
    return angle2vecs_V2(points2d, points3d, point_keys, limb_lengths, lengths_keys)


def left_knee_angle(points2d, points3d, limb_lengths):
    # Left hip, left knee, left ankle
    point_keys = ['p10', 'p12', 'p14']
    lengths_keys = ['hk', 'ka']
    return angle2vecs_V2(points2d, points3d, point_keys, limb_lengths, lengths_keys)
