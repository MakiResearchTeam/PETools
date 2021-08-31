import numpy as np
from math import sqrt

from numba import njit

EXP_SCALE = 2
EPSILONE = 1e-6


@njit
def ev_dist(x: float) -> float:
    return np.sqrt(np.sum(np.square(x)))


@njit
def calc_exp(one_p: float, center_p: float, max_radi_p: float, scale_eps: float = EXP_SCALE) -> float:
    hand2knee = one_p - center_p
    knee2foot = center_p - max_radi_p
    return np.exp(-scale_eps * (ev_dist(hand2knee) ** 2) / ((ev_dist(knee2foot)) ** 2 + EPSILONE))


@njit
def multi_sim(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, start_ind: int, out: np.ndarray):
    # Computes a cosine similarity between two vectors
    # (x1, y1) - first point
    # (x2, y2) - intermediate point
    # (x3, y3) - last point
    # --- First vector stats
    vec1x = x1 - x2
    vec1y = y1 - y2
    vec1norm = vec1x * vec1x + vec1y * vec1y
    if vec1norm < 1e-3:
        vec1x = 0.0
        vec1y = 0.0
        vec1norm = 1.0
    vec1norm = sqrt(vec1norm)

    out[start_ind] = vec1x / vec1norm
    out[start_ind + 1] = vec1y / vec1norm

    # --- Second vector stats
    vec2x = x3 - x2
    vec2y = y3 - y2
    vec2norm = vec2x * vec2x + vec2y * vec2y
    if vec2norm < 1e-3:
        vec2x = 0.0
        vec2y = 0.0
        vec2norm = 1.0
    vec2norm = sqrt(vec2norm)

    out[start_ind + 2] = vec2x / vec2norm
    out[start_ind + 3] = vec2y / vec2norm

    normalizer = vec1norm * vec2norm
    product = (vec1x * vec2x + vec1y * vec2y) / normalizer
    out[start_ind + 4] = product


@njit
def gen_f(points: np.ndarray, features: np.ndarray, point_triple_ids: np.ndarray, n_triples: int):
    for i in range(n_triples):
        ind1 = point_triple_ids[i, 0]
        ind2 = point_triple_ids[i, 1]
        ind3 = point_triple_ids[i, 2]
        multi_sim(
            points[ind1, 0], points[ind1, 1],
            points[ind2, 0], points[ind2, 1],
            points[ind3, 0], points[ind3, 1],
            i * 5,
            features
        )

    left_hand = points[8]
    right_hand = points[9]

    left_knee = points[12]
    right_knee = points[13]

    left_foot = points[14]
    right_foot = points[15]

    # Around knee
    # left to left
    features[-1] = calc_exp(left_hand, left_knee, left_foot)
    # right to right
    features[-2] = calc_exp(right_hand, right_knee, right_foot)
    # right to left
    features[-3] = calc_exp(right_hand, left_knee, left_foot)
    # left to right
    features[-4] = calc_exp(left_hand, right_knee, right_foot)
    # Around foot
    # left to left
    features[-5] = calc_exp(left_hand, left_foot, left_knee, 1.0)
    # right to right
    features[-6] = calc_exp(right_hand, right_foot, right_knee, 1.0)
    # right to left
    features[-7] = calc_exp(right_hand, left_foot, left_knee, 0.75)
    # left to right
    features[-8] = calc_exp(left_hand, right_foot, right_knee, 0.75)


class FeatureGenerator:

    def __init__(self, connectivity_list, n_points):
        # An idiot check
        for i in range(connectivity_list.shape[0]):
            assert connectivity_list[i, 0] < n_points
            assert connectivity_list[i, 1] < n_points

        # Make a list of points triples (vector pairs)
        self.point_triple_ids = np.array(connectivity_list, dtype=np.int32)
        self.n_triples = self.point_triple_ids.shape[0]

    def generate_features(self, points, features):
        # Compute distances
        gen_f(points, features, self.point_triple_ids, self.n_triples)
