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
    z = math.sqrt(diff if diff > 0. else 0.)
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


def compute_angle_2vec_V3(p1: list, p2: list, p3: list, p1p2_dist: float, p2p3_dist: float):
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
    v2 = compute_cuboid_diag(p2, p3, p2p3_dist, p2[2] > p3[2])
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


# Only for elbows
def angle2vecs_V3(points2d, points3d, points_keys, limb_lengths, limb_lengths_keys, left_shoulder_p, right_shoulder_p):
    p1 = extract_point(points2d, points3d, points_keys[0])
    p2 = extract_point(points2d, points3d, points_keys[1])
    p3 = extract_point(points2d, points3d, points_keys[2])
    p2p1_dist = limb_lengths[limb_lengths_keys[0]]
    p2p3_dist = limb_lengths[limb_lengths_keys[1]]

    # --- Correcting depth based on how shoulders are located in space
    def find_angle(v1, v2):
        # Finds a singed angle between to vectors.
        # If v2 is a rotated version of v1 by A degrees (A is positive), the A is returned,
        # otherwise -A is returned.
        norm_prod = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
        angle = np.degrees(np.arccos(norm_prod))
        return angle * -np.sign(np.cross(v1, v2))

    def make_rotmat(angle):
        angle = np.radians(angle)
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

    ls_p, rs_p = left_shoulder_p, right_shoulder_p
    ls_p, rs_p = np.array([ls_p[0], ls_p[2]]), np.array([rs_p[0], rs_p[2]])
    ls_rs_vec = rs_p - ls_p
    # Equivalent to rotating the vector by 90 degrees counterclockwise
    ls_rs_vec_normal = np.array([-ls_rs_vec[1], ls_rs_vec[0]])

    # Normal vector that points to the camera
    camera_normal = [0, -1]
    angle = find_angle(ls_rs_vec_normal, camera_normal)
    rotmat = make_rotmat(-angle)

    # Rotate points so that the shoulders normal has the same direction as the camera_normal
    middle_point = (ls_p + rs_p) / 2  # Every point is rotated around the middle point

    def compute_correct_z(init_point, middle_point, rotmat):
        init_point_vec = init_point - middle_point
        corrected_point_vec = rotmat.dot(init_point_vec)
        corrected_point = corrected_point_vec + middle_point
        return corrected_point[1]

    #p1[2] = compute_correct_z(np.array([p1[0], p1[2]]), middle_point, rotmat)
    #p2[2] = compute_correct_z(np.array([p2[0], p2[2]]), middle_point, rotmat)
    #p3[2] = compute_correct_z(np.array([p3[0], p3[2]]), middle_point, rotmat)

    # --- Magical points correction
    def euclid_dist(p1, p2):
        x = p1[0] - p2[0]
        y = p1[1] - p2[1]
        return math.sqrt(x ** 2 + y ** 2)

    def correct_points(p1, p2, dist_2d, dist_3d, ratio1, ratio2):
        # Pulls points together if they are close enough
        # and moves them away from each other if they are far apart enough
        ratio = dist_2d / dist_3d
        if ratio > ratio1:
            # Move point apart making the vectors longer
            x_shift = (p2[0] - p1[0]) * (1 - ratio) * 1.25  # magic numbers that work well
            y_shift = (p2[1] - p1[1]) * (1 - ratio) * 1.25
            p1[0] -= x_shift
            p1[1] -= y_shift
        elif ratio < ratio2:
            x_shift = (p2[0] - p1[0]) * ratio * 0.8
            y_shift = (p2[1] - p1[1]) * ratio * 0.8
            p1[0] += x_shift
            p1[1] += y_shift

    p2p1_2d_dist = euclid_dist(p2, p1)
    p2p3_2d_dist = euclid_dist(p2, p3)
    ratio1 = 0.85  # magic number that work well
    ratio2 = 0.25  # magic number that work well, -1 means the corresponding correction won't happen
    correct_points(p1, p2, p2p1_2d_dist, p2p1_dist, ratio1, ratio2)
    correct_points(p3, p2, p2p3_2d_dist, p2p3_dist, ratio1, ratio2)

    return compute_angle_2vec_V3(p1, p2, p3, p2p1_dist, p2p3_dist)


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
    # Left shoulder
    ls_p = extract_point(points2d, points3d, 'p4')
    # Left shoulder
    rs_p = extract_point(points2d, points3d, 'p5')
    return angle2vecs_V3(points2d, points3d, point_keys, limb_lengths, lengths_keys, ls_p, rs_p)


def left_elbow_angle(points2d, points3d, limb_lengths):
    # Left shoulder, left elbow, left wrist
    point_keys = ['p4', 'p6', 'p8']
    lengths_keys = ['se', 'ew']
    # Left shoulder
    ls_p = extract_point(points2d, points3d, 'p4')
    # Left shoulder
    rs_p = extract_point(points2d, points3d, 'p5')
    return angle2vecs_V3(points2d, points3d, point_keys, limb_lengths, lengths_keys, ls_p, rs_p)


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
