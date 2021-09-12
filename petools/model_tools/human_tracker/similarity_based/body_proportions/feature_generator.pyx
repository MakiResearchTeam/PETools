#cython: language_level=3

cimport numpy as cnp
import numpy as np
from libc.math cimport sqrt
cimport cython


cdef inline float dist(float x1, float y1, float x2, float y2):
    cdef float sq_dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    return sqrt(sq_dist)


cdef class FeatureGenerator:
    cdef:
        int[:, ::1] connectivity_list
        int n_points
        int n_connections
        float[::1] dist_buffer
        float eps

    def __cinit__(self, list connectivity_list, int n_points, float eps):
        """
        Parameters
        ----------
        connectivity_list : list
            List of pairs of indices that of points that denote one "limb".
        n_points : int
            For performing dummy check.
        eps : float
            If distance between 2 points

        Returns
        -------

        """
        self.connectivity_list = np.asarray(connectivity_list, dtype='int32')
        self.n_points = n_points
        self.n_connections = self.connectivity_list.shape[0]
        self.eps = eps
        # An idiot check
        for i in range(self.n_connections):
            assert self.connectivity_list[i, 0] < self.n_points
            assert self.connectivity_list[i, 1] < self.n_points

        self.dist_buffer = np.empty(shape=[self.n_connections], dtype='float32')

    @cython.wraparound(False)
    cpdef cnp.ndarray[float, ndim=2] generate_features(self, cnp.ndarray[float, ndim=2] points):
        """
        Computes a matrix F of features, where F[i, j] equals to the ratio of ith limb to jth limb.
         
        Parameters
        ----------
        points : np.ndarray[float, ndim=2]
            An array of points to compute the feature matrix for.

        Returns
        -------
        feature_matrix : np.ndarray[float, ndim=2]
            A matrix of shape [n_limbs, n_limbs].
        """
        cdef:
            size_t i = 0, j = 0, j_inv = 0
            int ind1, ind2
            float eps = self.eps
            cnp.ndarray[float, ndim=2] feat_mat = np.empty(shape=[self.n_connections, self.n_connections], dtype='float32')

        # Compute distances
        for i in range(self.n_connections):
            ind1 = self.connectivity_list[i, 0]
            ind2 = self.connectivity_list[i, 1]
            self.dist_buffer[i] = dist(points[ind1, 0], points[ind1, 1], points[ind2, 0], points[ind2, 1])

            if self.dist_buffer[i] < eps:
                self.dist_buffer[i] = eps

        # Compute features
        # F[i, j] = dist[i] / dist[j]
        for i in range(self.n_connections):
            for j in range(self.n_connections):
                feat_mat[i, j] = self.dist_buffer[i] / self.dist_buffer[j]

        return feat_mat
