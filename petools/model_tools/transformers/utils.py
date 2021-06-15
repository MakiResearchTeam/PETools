CONVERT_STATS_3D = '3d_converter_stats'

MEAN_2D_V1 = 'mean_2d_v1.npy'
MEAN_2D_V2 = 'mean_2d.npy'
MEAN_2D = MEAN_2D_V2

MEAN_3D_V1 = 'mean_3d_v1.npy'
MEAN_3D_V2 = 'mean_3d.npy'
MEAN_3D = MEAN_3D_V2

STD_2D_V1 = 'std_2d_v1.npy'
STD_2D_V2 = 'std_2d.npy'
STD_2D = STD_2D_V2

STD_3D_V1 = 'std_3d_v1.npy'
STD_3D_V2 = 'std_3d.npy'
STD_3D = STD_3D_V2

# Map production keypoints (predicted from NN)
# To keypoints of Human3.6M dataset for suitable input into 3d converter
PRODUCTION_TO_HUMAN36_V1 = [
    # LOWER BODY
    # middle hip
    [22, 0],
    # right hip
    [11, 1],
    # left hip
    [10, 4],

    # right knee
    [13, 2],
    # left knee
    [12, 5],

    # right foot
    [15, 3],
    # left foot
    [14, 6],

    # UPPER BODY
    # center
    [0, 7],
    [1, 8],
    # left shoulder
    [4, 11],
    # right shoulder
    [5, 14],

    # neck
    [2, 9],
    # head
    [3, 10],

    # HANDS
    # left elbow
    [6, 12],
    # right elbow
    [7, 15],

    # left wrist
    [8, 13],
    # right wrist
    [9, 16]
]
PRODUCTION_TO_HUMAN36_V2 = [
    # LOWER BODY
    # middle hip
    [22, 0],            # 1
    # right hip
    [11, 1],            # 2
    # left hip
    [10, 5],            # 3

    # right knee
    [13, 2],            # 4
    # left knee
    [12, 6],            # 5

    # right foot
    [15, 3],           # 6
    # left foot
    [14, 7],           # 7

    # right foot middle of fingers
    [17, 4],           # 8
    # left foot middle of fingers
    [16, 8],           # 9

    # UPPER BODY
    # center
    [0, 9],            # 10
    [1, 10],           # 11
    # left shoulder
    [4, 13],           # 12
    # right shoulder
    [5, 19],           # 13

    # neck
    [2, 11],           # 14
    # head
    [3, 12],           # 15

    # HANDS
    # left elbow
    [6, 14],           # 16
    # right elbow
    [7, 20],           # 17

    # left wrist
    [8, 15],           # 18
    # right wrist
    [9, 21],           # 19

    # left hand fingers
    [18, 17],          # 20
    [19, 18],          # 21
    # right hand fingers
    [20, 22],          # 22
    [21, 23],          # 23
]
PRODUCTION_TO_HUMAN36 = PRODUCTION_TO_HUMAN36_V2

# Map keypoints3d from Human3.6M dataset to production keypoints3d (order)
H32_TO_HPROD_3D_V1 = [
    # (h32, h16)
    (0, 0),  # 1
    (1, 1),  # 2
    (2, 2),  # 3

    (3, 3),  # 4
    (6, 4),  # 5
    (7, 5),  # 6

    (8, 6),  # 7
    (12, 7),  # 8
    (13, 8),  # 9

    (15, 9),  # 10
    (17, 10),  # 11
    (18, 11),  # 12

    (19, 12),  # 13
    (25, 13),  # 14
    (26, 14),  # 15
    (27, 15),  # 16
]

H32_TO_HPROD_3D_V2 = [
    # (h32, h22)
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (5, 4),

    (6, 5),
    (7, 6),
    (8, 7),
    (10, 8),

    (12, 9),
    (13, 10),
    (15, 12),

    (17, 13),
    (18, 14),
    (19, 15),
    (21, 17),
    (22, 18),

    (25, 15),
    (26, 16),
    (27, 17),
    (29, 22),
    (30, 23),
]
H32_TO_HPROD_3D = H32_TO_HPROD_3D_V2

# Map predicted 3d points into the Human3.6M order of keypoints
# 16, wo hip
INDICES3DTO32POINTS_V1 = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
# 22, wo hip
INDICES3DTO32POINTS_V2 = [1, 2, 3, 5, 6, 7, 8, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
INDICES3DTO32POINTS = INDICES3DTO32POINTS_V2

NUM_PROD_POINTS = 23

NECK_POINT_V1 = 9
NECK_POINT_V2 = -1
NECK_POINT = NECK_POINT_V2

H3P6_2D_NUM_V1 = 17
H3P6_2D_NUM_V2 = 23
H3P6_2D_NUM = H3P6_2D_NUM_V2

H3P6_3D_NUM_V1 = 32
H3P6_3D_NUM_V2 = 44
H3P6_3D_NUM = H3P6_3D_NUM_V2


def init_selector_v1():
    # The input to the network does not include neck point
    select_2d = [True] * H3P6_2D_NUM_V1
    select_2d[NECK_POINT_V1] = False
    return select_2d


def init_selector_v2():
    # The input to the network does not include neck point
    select_2d = [True] * H3P6_2D_NUM_V2
    select_2d[NECK_POINT_V2] = False
    return select_2d


def init_selector():
    return init_selector_v2()

