import json
import os
import time

import cv2
import numpy as np
import tensorflow as tf

from petools.core import modify_humans, load_graph_def, PosePredictorInterface
from petools.tools.estimate_tools.algorithm_connect_skelet import estimate_paf, merge_similar_skelets
from petools.tools.utils import CAFFE, preprocess_input, scale_predicted_kp
from petools.tools.utils.video_tools import scales_image_single_dim_keep_dims
from .utils import INPUT_TENSOR, IND_TENSOR, PAF_TENSOR, PEAKS_SCORE_TENSOR


class PosePredictor(PosePredictorInterface):
    """
    PosePredictor - wrapper of PEModel from MakiPoseNet
    Contains main tools for drawing skeletons and predict them.

    """
    INPUT_NAME = INPUT_TENSOR
    IND_TENSOR_NAME = IND_TENSOR
    PAF_NAME = PAF_TENSOR
    PEAKS_SCORE_NAME = PEAKS_SCORE_TENSOR
    UPSAMPLE_SIZE = 'upsample_size'

    def __init__(
            self,
            path_to_pb: str,
            path_to_config: str,
            min_h=320,
            norm_mode=CAFFE,
            gpu_id=0
    ):
        """
        Create Pose Predictor wrapper of PEModel
        Which contains main function in order to estimate poses and draw skeletons

        Parameters
        ----------
        path_to_pb : str
            Path to pb file which contains estimate_tools obj,
            Example: "/home/user/estimate_tools.pb"
        path_to_config : str
            Path to config for pb file,
            This config contains of input/output information from estimate_tools, in order to get proper tensors
        min_h : tuple
            H_min
        norm_mode : str
            Mode to normalize input images, default CAFFE, i.e. image will be normalized according to ImageNet dataset
        gpu_id : int or str
            Number of GPU, which must be used to run estimate_tools on it,
            If CPU is needed - enter any symbol (expect digits), for example: ";"

        """

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.__min_h = min_h
        self.__norm_mode = norm_mode
        self.__path_to_tb = path_to_pb
        self.__path_to_config = path_to_config
        self._init_model()

    def _init_model(self):
        """
        Loads config and the estimate_tools's weights

        """
        with open(self.__path_to_config, 'r') as f:
            config = json.load(f)

        self.__graph_def = load_graph_def(self.__path_to_tb)
        self.__in_x = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='in_x')
        self.__upsample_size = tf.placeholder(dtype=tf.int32, shape=(2), name='upsample')
        self.__resized_paf, self.__indices, self.__peaks_score = tf.import_graph_def(
            self.__graph_def,
            input_map={
                config[PosePredictor.INPUT_NAME]: self.__in_x,
                PosePredictor.UPSAMPLE_SIZE: self.__upsample_size
            },
            return_elements=[
                config[PosePredictor.PAF_NAME],
                config[PosePredictor.IND_TENSOR_NAME],
                config[PosePredictor.PEAKS_SCORE_NAME]
            ]
        )
        self.__sess = tf.Session()

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

        return (new_h, new_w), self.SCALE - new_w % self.SCALE

    def predict(self, image: np.ndarray):

        """
        Estimate poses on single image

        Parameters
        ----------
        image : np.ndarray
            Input image, with shape (H, W, 3): H - Height, W - Width (H and W can have any values)
            For mose models - input image must be in bgr order

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
            Where PosePredictor.HUMANS and PosePredictor.TIME - are strings ('humans' and 'time')
        """
        # Get final image size and padding value
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
        # Measure time of prediction
        start_time = time.time()
        humans = self._predict(norm_img)[0]
        end_time = time.time() - start_time

        # Scale prediction to original image
        scale_predicted_kp(
            predictions=[humans],
            model_size=(new_h, new_w),
            source_size=image.shape[:-1]
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
        resize_to = norm_img[0].shape[:2]

        batched_paf, indices, peaks = self.__sess.run(
            [self.__resized_paf, self.__indices, self.__peaks_score],
            feed_dict={
                self.__in_x: norm_img,
                self.__upsample_size: resize_to
            }
        )

        return [
            merge_similar_skelets(estimate_paf(
                peaks=peaks.astype(np.float32, copy=False),
                indices=indices.astype(np.int32, copy=False),
                paf_mat=batched_paf[0]
            ))
        ]


if __name__ == '__main__':
    print(PosePredictor.HUMANS)

