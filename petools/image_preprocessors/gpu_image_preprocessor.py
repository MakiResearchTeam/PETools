import cv2
import numpy as np

from petools.core import ImagePreprocessor
from petools.tools.utils.video_tools import scales_image_single_dim_keep_dims
from petools.tools.utils import CAFFE, preprocess_input, scale_predicted_kp


class GpuImagePreprocessor(ImagePreprocessor):
    def __init__(self, h, w, scale, w_by_h, norm_mode):
        self.__min_h = h
        self.__max_w = w
        self.__scale = scale
        self.__w_by_h = w_by_h
        self.__norm_mode = norm_mode
        self.__resize_to = np.array([h, w]).astype(np.int32)
        self._recent_input_img_size = None
        self._saved_img_settings = None
        self._saved_padding_h = None
        self._saved_padding_w = None

    def __call__(self, image):
        (new_h, new_w), padding = self.__get_image_info(image.shape[:-1])
        resized_img = cv2.resize(image, (new_w, new_h))
        if padding:
            # Pad image with zeros,
            # In order to image be divided by PosePredictor.SCALE (in most cases equal to 8) without reminder
            single_img_input = np.zeros((new_h, new_w + padding, 3)).astype(np.uint8, copy=False)
            single_img_input[:, :resized_img.shape[1]] = resized_img
        else:
            single_img_input = resized_img

        # Add batch dimension
        img = np.expand_dims(single_img_input, axis=0).astype(np.float32, copy=False)
        # Normalize image
        norm_img = preprocess_input(img, mode=self.__norm_mode).astype(np.float32, copy=False)
        return norm_img

    def __get_image_info(self, image_size: list):
        """

        Parameters
        ----------
        image_size : list
            (H, W) of input image

        Returns
        -------
        (H, W) : tuple
            Height and Width of final input image into estimate_tools
        padding : int
            Number of padding need to be added to W, in order to be divided by 8 without remains

        """
        scale_x, scale_y = scales_image_single_dim_keep_dims(
            image_size=image_size,
            resize_to=self.__min_h
        )
        new_w, new_h = round(scale_x * image_size[1]), round(scale_y * image_size[0])

        return (new_h, new_w), self.__scale - new_w % self.__scale
