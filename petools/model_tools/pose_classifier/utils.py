CURRENT_VERSION = 'v1'

CLASSIFIER_STATS = 'classifier_stats'

MEAN_2D_V1 = f'mean_2d_{CURRENT_VERSION}.npy'
MEAN_2D = MEAN_2D_V1


STD_2D_V1 = f'std_2d_{CURRENT_VERSION}.npy'
STD_2D = STD_2D_V1

NUM_C = 5
DIMS = 11

INPUT_SHAPE = DIMS * NUM_C


conlist = [
    # torso
    [0, 1],  # center - chest
    [1, 2],  # chest - neck
    # [2, 3], # neck - head
    [0, 22],  # center - center hip

    # left hand
    [2, 4],  # neck - left shoulder
    [4, 6],  # left shoulder - left elbow
    [6, 8],  # left elbow - left wrist
    # [8, 18],# left wrist - left big finger
    # [8, 19],# left wrist - left small finger

    # right hand
    [2, 5],  # neck - right shoulder
    [5, 7],  # right shoulder - right elbow
    [7, 9],  # right elbow - right wrist
    # [9, 20],# right wrist - right big finger
    # [9, 21],# right wrist - right small finger

    # left leg
    [22, 10],  # center hip - left hip
    [10, 12],  # left hip - left knee
    [12, 14],  # left knee - left ankle
    # [14, 16],# left ankle - left leg finger

    # right leg
    [22, 11],  # center hip - right hip
    [11, 13],  # right hip - right knee
    [13, 15],  # right knee - right ankle
    # [15, 17],# right ankle - right leg finger
]

