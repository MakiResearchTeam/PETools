import cv2


class TextDrawConfig:
    """
    Used by the `_draw_human` function to get default arguments' values.
    """
    POSE_NAME_FONT = cv2.FONT_HERSHEY_SIMPLEX
    POSE_NAME_FONT_SCALE = 2
    POSE_NAME_FONT_COLOR = (0, 0, 255)
    POSE_NAME_FONT_THICKNESS = 6

    POSE_CONF_FONT = cv2.FONT_HERSHEY_SIMPLEX
    POSE_CONF_FONT_COLOR = (0, 0, 255)
    POSE_CONF_FONT_SCALE = 2
    POSE_CONF_FONT_THICKNESS = 6
