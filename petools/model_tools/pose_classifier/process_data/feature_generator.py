from ..utils import NUM_C

import numpy as np
from math import sqrt

from numba import njit


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
        # Hip dist to hip-neck
        if np.prod(points[11]) < 1e-3 or np.prod(points[10]) < 1e-3 or \
            np.prod(points[22]) < 1e-3 or np.prod(points[2]) < 1e-3 or np.prod(points[22] - points[2]) < 1e-3:
            dist = np.zeros((2,), dtype=np.float32)
        else:
            ev_dist = lambda x: np.sqrt(np.sum(np.square(x)))
            dist = ev_dist(points[11] - points[10]) / ev_dist(points[22] - points[2])
        features[-1] = dist[0] # x
        features[-2] = dist[1] # y

        dist = np.sign(points[2] - points[22])
        features[-3] = dist[1] # y
