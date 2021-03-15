import numpy as np

from petools.core import PosePredictorInterface
from petools.tools.estimate_tools.constants import CONNECT_KP
from .visualize_tools import draw_skeleton


def draw_skeletons_on_image(image: np.ndarray, predictions: dict, color=(255, 0, 0), thick=3):
    """
    Draw skeletons from `preidctions` on certain `image`
    With parameters such as color and thick of the line

    Parameters
    ----------
    image : np.ndarray
        The image on which detection was performed
    predictions : dict
        Prediction on `image` from this class and method `predict`
    color : tuple
        Color of the line,
        By default equal to (255, 0, 0) - i.e. red line
    thick : int
        Thick of the line, by default equal to 3, in most cases this value is enough

    Returns
    -------
    np.ndarray
        Image with skeletons on it

    """
    predictions_humans = predictions[PosePredictorInterface.HUMANS]
    humans = [list(single_h.values()) for single_h in predictions_humans]
    return draw_skeleton(image.copy(), humans, connect_indexes=CONNECT_KP, color=color, thickness=thick)

