from numba import njit
import numpy as np


@njit
def norm2d_njit(human: np.ndarray, mean2d: np.ndarray, std2d: np.ndarray) -> np.ndarray:
    return (human.reshape(-1) - mean2d) / std2d


@njit
def denorm2d_njit(human: np.ndarray, mean2d: np.ndarray, std2d: np.ndarray) -> np.ndarray:
    return human.reshape(-1) * std2d + mean2d


@njit
def norm3d_njit(human: np.ndarray, mean3d: np.ndarray, std3d: np.ndarray) -> np.ndarray:
    return (human.reshape(-1) - mean3d) / std3d


@njit
def denorm3d_njit(human: np.ndarray, mean3d: np.ndarray, std3d: np.ndarray) -> np.ndarray:
    return human.reshape(-1) * std3d + mean3d


@njit
def denorm3d_pca_njit(human: np.ndarray, pca_matrix: np.ndarray) -> np.ndarray:
    human = np.dot(human, pca_matrix)
    return human

