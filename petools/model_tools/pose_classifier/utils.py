CURRENT_VERSION = 'v1'

CLASSIFIER_STATS = 'classifier_stats'

MEAN_2D_V1 = f'mean_2d_{CURRENT_VERSION}.npy'
MEAN_2D = MEAN_2D_V1


STD_2D_V1 = f'std_2d_{CURRENT_VERSION}.npy'
STD_2D = STD_2D_V1


conlist = [
    [0, 1, 2],  # center body - chest - neck
    [1, 2, 4],  # chest - neck - right shoulder
    [1, 2, 5],  # chest - neck - left shoulder
    [0, 22, 10],  # center body - hip - right hip
    [0, 22, 11],  # center body - hip - left hip
    [2, 4, 6],  # neck - right shoulder - elbow right hand
    [4, 6, 8],  # right shoulder - elbow right hand - right hand
    [2, 5, 7],  # neck - left shoulder - elbow left hand
    [5, 7, 9],  # left shoulder - elbow left hand - left hand
    [22, 10, 12],  # hip - right hip - right knee
    [10, 12, 14],  # right hip - right knee - right foot
    [22, 11, 13],  # hip - left hip - left knee
    [11, 13, 15],  # left hip - left knee - left foot

    [11, 5, 7],  # left hip - left shoulder - left elbow
    [11, 7, 9],  # left hip - left elbow - left hand
    [10, 4, 6],  # right hip - right shoulder - right elbow
    [10, 6, 8],  # right hip - right elbow - right hand
]


NUM_C = 5
DIMS = len(conlist)
ADDITIONAL_INFO = 3
INPUT_SHAPE = DIMS * NUM_C + ADDITIONAL_INFO
