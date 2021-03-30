import json
import os
import time

import cv2
import numpy as np
import tensorflow as tf

from petools.core import PosePredictorInterface
from petools.tools.estimate_tools.skelet_builder import SkeletBuilder
from petools.tools.utils import CAFFE, preprocess_input, scale_predicted_kp
from petools.tools.utils.video_tools import scales_image_single_dim_keep_dims
from petools.tools.utils.nns_tools.modify_skeleton import modify_humans
from .utils import IMAGE_INPUT_SIZE
from .cpu_postprocess_np_part import CPUOptimizedPostProcessNPPart


class PosePredictor(PosePredictorInterface):
    """
    PosePredictor - wrapper of PEModel from MakiPoseNet
    Contains main tools for drawing skeletons and predict them.

    """
    W_BY_H = 1128.0 / 1920.0

    def __init__(
            self,
            path_to_tflite: str,
            path_to_config: str,
            norm_mode=CAFFE,
            gpu_id=';',
            num_threads=None,
            kp_scale_end=2
    ):
        """
        Create Pose Predictor wrapper of PEModel
        Which contains main function in order to estimate poses and draw skeletons

        Parameters
        ----------
        path_to_tflite : str
            Path to tflite file which contains estimate_tools obj,
            Example: "/home/user/estimate_tools.tflite"
        path_to_config : str
            Path to config for pb file,
            This config contains of input/output information from estimate_tools, in order to get proper tensors
        norm_mode : str
            Mode to normalize input images, default TF, i.e. image will be in range (-1, 1)
        gpu_id : int or str
            Number of GPU, which must be used to run estimate_tools on it,
            If CPU is needed - enter any symbol (expect digits), for example: ";",
            By default CPU is used

        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.__norm_mode = norm_mode
        self.__path_to_tb = path_to_tflite
        self.__path_to_config = path_to_config
        self.__num_threads = num_threads
        self._kp_scale_end = kp_scale_end
        self._init_model()

    def _init_model(self):
        """
        Loads config and the estimate_tools's weights

        """
        with open(self.__path_to_config, 'r') as f:
            config = json.load(f)
        H, W = config[IMAGE_INPUT_SIZE]
        self.__min_h = H
        self.__max_w = W
        self.__resize_to = np.array([H, W]).astype(np.int32)
        self._recent_input_img_size = None
        self._saved_img_settings = None
        self._saved_padding_h = None
        self._saved_padding_w = None

        interpreter = tf.lite.Interpreter(model_path=str(self.__path_to_tb), num_threads=self.__num_threads)
        interpreter.allocate_tensors()
        self.__interpreter = interpreter
        self.__in_x = interpreter.get_input_details()[0]["index"]
        #self.__upsample_size = interpreter.get_input_details()[1]["index"]

        self.__paf_tensor = interpreter.get_output_details()[0]["index"]
        self.__heatmap_tensor = interpreter.get_output_details()[1]["index"]

        self._postprocess_np = CPUOptimizedPostProcessNPPart(
            resize_to=(H, W),
            upsample_heatmap=False,
            kp_scale_end=self._kp_scale_end
        )

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
                (self._recent_input_img_size[0] != image_size[0] and self._recent_input_img_size[1] != image_size[1]):
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
                recalc_w = int(image_size[1] * PosePredictor.W_BY_H)
                new_image_size = (
                    recalc_w + (PosePredictor.SCALE + recalc_w % PosePredictor.SCALE) - PosePredictor.SCALE,
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

    def predict(self, image: np.ndarray):
        """
        Estimate poses on single image

        Parameters
        ----------
        image : np.ndarray
            Input image, with shape (H, W, 3): H - Height, W - Width (H and W can have any values)
            For most of models - input image must be in bgr order

        Returns
        -------
        dict
            Single predictions as dict object contains of:
            {
                PosePredictor.HUMANS: [
                        [
                            [h1_x_1, h1_y_1, h1_v_1],
                            [h1_x_2, h1_y_2, h1_v_2],
                            ...
                            [h1_x_n, h1_y_n, h1_v_n],
                        ],

                        [
                            [h2_x_1, h2_y_1, h2_v_1],
                            [h2_x_2, h2_y_2, h2_v_2],
                            ...
                            [h2_x_n, h2_y_n, h2_v_n],
                        ]

                        ...
                        ...

                        [
                            [hN_x_1, hN_y_1, hN_v_1],
                            [hN_x_2, hN_y_2, hN_v_2],
                            ...
                            [hN_x_n, hN_y_n, hN_v_n],
                        ]
                ],
                PosePredictor.TIME: some_float_number
            }

        """
        original_in_size = image.shape[:-1]
        # Get final image size and padding value
        (new_h, new_w), padding, padding_h_before_resize = self.__get_image_info(image.shape[:-1])
        # Padding image by H axis with zeros
        # in order to be more suitable size after resize to new_w and new_h
        if padding_h_before_resize is not None:
            # Pad image with zeros,
            if self._saved_padding_h is None or \
                    self._saved_padding_h.shape[0] != (image.shape[0]+padding_h_before_resize) or \
                    self._saved_padding_h.shape[1] != image.shape[1] or \
                    self._saved_padding_h.shape[2] != image.shape[2]:
                padding_image = np.zeros(
                    (image.shape[0]+padding_h_before_resize, image.shape[1], image.shape[2]),
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
        # Measure time of prediction
        start_time = time.time()
        humans = self._predict(norm_img)
        end_time = time.time() - start_time

        # Scale prediction to original image
        up_h = int(new_h - (padding_h_before_resize * (resized_img.shape[0] / image.shape[0])))
        scale_predicted_kp(
            predictions=[humans],
            model_size=(up_h, new_w),
            source_size=original_in_size
        )

        updated_humans = modify_humans(humans)
        return {
            PosePredictor.HUMANS: [
                dict(list(map(lambda indx, in_x: (indx, in_x), range(PosePredictor.NUM_KEYPOINTS), single_human)))
                for single_human in updated_humans
            ],
            PosePredictor.TIME: end_time
        }

    def _predict(self, norm_img):
        """
        Imitates PEModel's predict.

        Parameters
        ----------
        norm_img : ndarray
            Normalized image according with the estimate_tools's needs.

        Returns
        -------
        list
            List of predictions to each input image.
            Single element of this list is a List of classes Human which were detected.
        """
        interpreter = self.__interpreter
        interpreter.set_tensor(self.__in_x, norm_img)
        #interpreter.set_tensor(self.__upsample_size, self.__resize_to)
        # Run estimate_tools
        interpreter.invoke()

        paf_pr, smoothed_heatmap_pr = (
            interpreter.get_tensor(self.__paf_tensor),
            interpreter.get_tensor(self.__heatmap_tensor)
        )
        upsample_paf, indices, peaks = self._postprocess_np.process(heatmap=smoothed_heatmap_pr, paf=paf_pr)
        return SkeletBuilder.get_humans_by_PIF(peaks=peaks, indices=indices, paf_mat=upsample_paf)
