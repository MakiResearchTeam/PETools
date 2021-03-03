import cv2
from petools.core import PosePredictorInterface


class SkeletonDrawer:
    def __init__(self, video_path, fps=20, color=(255, 0, 0)):
        """

        Parameters
        ----------
        video_path
        fps
        color

        """
        self._video_path = video_path
        self._fps = fps
        self._color = color
        self._video = None

    def _init(self, frame_size):
        height, width = frame_size
        self._video = cv2.VideoWriter(
            self._video_path, cv2.VideoWriter_fourcc(*'mp4v'), self._fps,
            (width, height)
        )

    def write(self, images, predictions):
        """
        Draws skeletons on the `images` according to the give `predictions`.

        Parameters
        ----------
        images : list
            List of images (ndarrays).
        predictions : list
            List of lists that contain instances of class Human.
        """
        if self._video is None:
            h, w, c = images[0].shape
            self._init((h, w))

        for image, prediction in zip(images, predictions):
            image = PosePredictorInterface.draw(image, prediction, self._color)
            self._video.write(image)

    def release(self):
        self._video.release()
