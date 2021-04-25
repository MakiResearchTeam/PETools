from ..core import ImagePreprocessor
from petools.tools.utils.video_tools import scales_image_single_dim_keep_dims
from petools.tools.utils import CAFFE, preprocess_input, scale_predicted_kp

import cv2
import numpy as np


class CpuImagePreprocessor(ImagePreprocessor):
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

    def preprocess(self, image):
        original_in_size = image.shape[:-1]
        # Get final image size and padding value
        (new_h, new_w), padding, padding_h_before_resize = self.__get_image_info(image.shape[:-1])
        # Padding image by H axis with zeros
        # in order to be more suitable size after resize to new_w and new_h
        if padding_h_before_resize is not None:
            # Pad image with zeros,
            if self._saved_padding_h is None or \
                    self._saved_padding_h.shape[0] != (image.shape[0] + padding_h_before_resize) or \
                    self._saved_padding_h.shape[1] != image.shape[1] or \
                    self._saved_padding_h.shape[2] != image.shape[2]:
                padding_image = np.zeros(
                    (image.shape[0] + padding_h_before_resize, image.shape[1], image.shape[2]),
                    dtype=np.uint8
                )
                self._saved_padding_h = padding_image
            else:
                padding_image = self._saved_padding_h
            padding_image[:image.shape[0]] = image
            image = padding_image
        else:
            padding_h_before_resize = 0
        # Apply resize
        resized_img = cv2.resize(image, (new_w, new_h))
        # Pad image with zeros,
        # In order to image be divided by PosePredictor.SCALE (in most cases equal to 8) without reminder
        if padding:
            # Pad image with zeros,
            if self._saved_padding_w is None or \
                    self._saved_padding_w.shape[0] != new_h or \
                    self._saved_padding_w.shape[1] != (new_w + padding) or \
                    self._saved_padding_w.shape[2] != 3:
                single_img_input = np.zeros((new_h, new_w + padding, 3), dtype=np.uint8)
                self._saved_padding_w = single_img_input
            else:
                single_img_input = self._saved_padding_w
            single_img_input[:, :resized_img.shape[1]] = resized_img
        else:
            single_img_input = resized_img

        # Add batch dimension
        img = np.expand_dims(single_img_input, axis=0)
        # Normalize image
        norm_img = preprocess_input(img, mode=self.__norm_mode)
        up_h = int(new_h - (padding_h_before_resize * (resized_img.shape[0] / image.shape[0])))
        return norm_img, up_h, new_w, original_in_size

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
        padding_h_before_resize : int
            Padding h before resize operation, if equal to None,
            i.e. this operation (padding) is not required

        """
        if self._recent_input_img_size is None or \
                self._recent_input_img_size[0] != image_size[0] or \
                self._recent_input_img_size[1] != image_size[1]:
            padding_h_before_resize = None
            self._recent_input_img_size = image_size

            scale_x, scale_y = scales_image_single_dim_keep_dims(
                image_size=image_size,
                resize_to=self.__min_h
            )
            new_w, new_h = round(scale_x * image_size[1]), round(scale_y * image_size[0])

            if self.__max_w - new_w <= 0:
                # Find new value for H which is more suitable in order to calculate lower image
                # And give that to model
                recalc_w = int(image_size[1] * self.__w_by_h)
                new_image_size = (
                    recalc_w + (self.__scale + recalc_w % self.__scale) - self.__scale,
                    image_size[1]
                )
                # We must add zeros by H dimension in original image
                padding_h_before_resize = new_image_size[0] - image_size[0]
                # Again calculate resize scales and calculate new_w and new_h for resize operation
                scale_x, scale_y = scales_image_single_dim_keep_dims(
                    image_size=new_image_size,
                    resize_to=self.__min_h
                )
                new_w, new_h = round(scale_x * new_image_size[1]), round(scale_y * new_image_size[0])

            self._saved_img_settings = ((new_h, new_w), self.__max_w - new_w, padding_h_before_resize)
        return self._saved_img_settings
