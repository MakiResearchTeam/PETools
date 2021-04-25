import json
import os
import time

import cv2
import numpy as np
import tensorflow as tf

from petools.core import PosePredictorInterface
from petools.tools.estimate_tools.skelet_builder import SkeletBuilder
from petools.tools.utils import CAFFE, preprocess_input, scale_predicted_kp

from petools.tools.utils.nns_tools.modify_skeleton import modify_humans
from .utils import IMAGE_INPUT_SIZE
from .cpu_postprocess_np_part import CPUOptimizedPostProcessNPPart
from .cpu_image_preprocessor import CpuImagePreprocessor
from .cpu_model import CpuModel


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

        self.__image_preprocessor = CpuImagePreprocessor(
            h=H,
            w=W,
            scale=PosePredictor.SCALE,
            w_by_h=PosePredictor.W_BY_H,
            norm_mode=self.__norm_mode
        )
        self._model = CpuModel(
            tflite_file=self.__path_to_tb,
            num_threads=self.__num_threads
        )
        self._postprocess_np = CPUOptimizedPostProcessNPPart(
            resize_to=(H, W),
            upsample_heatmap=False,
            kp_scale_end=self._kp_scale_end
        )

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
                "humans": [
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
                "time": some_float_number
            }

        """
        # 1. Preprocess image before feeding into the NN
        start_time = time.time()
        norm_img, up_h, new_w, original_in_size = self.__image_preprocessor.preprocess(image)
        # 2. Feed the image into the NN and get PAF and heatmap tensors
        paf_pr, smoothed_heatmap_pr = self._model.predict(norm_img)
        # 3. Post process PAF and heatmap
        upsample_paf, indices, peaks = self._postprocess_np.process(heatmap=smoothed_heatmap_pr, paf=paf_pr)
        # 4. Build skeletons based off postprocessing results
        humans = SkeletBuilder.get_humans_by_PIF(peaks=peaks, indices=indices, paf_mat=upsample_paf)
        # 5. Scale skeletons' coordinates to the original image size
        scale_predicted_kp(
            predictions=[humans],
            model_size=(up_h, new_w),
            source_size=original_in_size
        )
        # 6. Perform additional correction
        updated_humans = modify_humans(humans)
        end_time = time.time() - start_time
        return {
            PosePredictor.HUMANS: [
                dict(list(map(lambda indx, in_x: (indx, in_x), range(PosePredictor.NUM_KEYPOINTS), single_human)))
                for single_human in updated_humans
            ],
            PosePredictor.TIME: end_time
        }
