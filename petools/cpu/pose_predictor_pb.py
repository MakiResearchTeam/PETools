import json
import os
import time
import pathlib
import tensorflow as tf
import numpy as np

from petools.core import PosePredictorInterface
from petools.tools.estimate_tools.skelet_builder import SkeletBuilder
from petools.tools.utils import CAFFE, scale_predicted_kp
from petools.tools.utils.nns_tools.modify_skeleton import modify_humans
from petools.tools.estimate_tools import Human

from .utils import IMAGE_INPUT_SIZE, INPUT_TENSOR, PAF_TENSOR, SMOOTHED_HEATMAP_TENSOR
from .cpu_postprocess_np_part import CPUOptimizedPostProcessNPPart
from ..image_preprocessors import CpuImagePreprocessor
from .cpu_model_pb import CpuModelPB
from .cpu_converter_3d import CpuConverter3D


class PosePredictor(PosePredictorInterface):
    """
    PosePredictor - wrapper of PEModel from MakiPoseNet
    Contains main tools for drawing skeletons and predict them.

    """
    W_BY_H = 1128.0 / 1920.0

    def __init__(
            self,
            path_to_pb: str,
            path_to_config: str,
            path_to_tflite_3d: str = None,
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
        path_to_pb : str
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
        self._norm_mode = norm_mode
        self._path_to_pb = path_to_pb
        self._path_to_config = path_to_config
        self._path_to_tflite_3d = path_to_tflite_3d
        self._num_threads = num_threads
        self._kp_scale_end = kp_scale_end
        self._init_model()

    def _init_model(self):
        """
        Loads config and the estimate_tools's weights

        """
        with open(self._path_to_config, 'r') as f:
            config = json.load(f)
        H, W = config[IMAGE_INPUT_SIZE]

        self.__sess = tf.Session()

        self.__image_preprocessor = CpuImagePreprocessor(
            h=H,
            w=W,
            scale=PosePredictor.SCALE,
            w_by_h=PosePredictor.W_BY_H,
            norm_mode=self._norm_mode
        )
        self._model = CpuModelPB(
            pb_path=self._path_to_pb,
            input_name=config[INPUT_TENSOR],
            paf_name=config[PAF_TENSOR],
            heatmap_name=config[SMOOTHED_HEATMAP_TENSOR],
            session=self.__sess
        )
        self._postprocess_np = CPUOptimizedPostProcessNPPart(
            resize_to=(H, W),
            upsample_heatmap=False,
            kp_scale_end=self._kp_scale_end
        )

        if self._path_to_tflite_3d is not None:
            # --- INIT CONVERTER3D
            file_path = os.path.abspath(__file__)
            dir_path = pathlib.Path(file_path).parent
            data_stats_dir = os.path.join(dir_path, '3d_converter_stats')
            mean_path = os.path.join(data_stats_dir, 'mean_2d.npy')
            assert os.path.isfile(mean_path), f"Could not find mean_2d.npy in {mean_path}."
            mean_2d = np.load(mean_path)

            std_path = os.path.join(data_stats_dir, 'std_2d.npy')
            assert os.path.isfile(std_path), f"Could not find std_2d.npy in {std_path}."
            std_2d = np.load(std_path)

            mean_path = os.path.join(data_stats_dir, 'mean_3d.npy')
            assert os.path.isfile(mean_path), f"Could not find mean_3d.npy in {mean_path}."
            mean_3d = np.load(mean_path)

            std_path = os.path.join(data_stats_dir, 'std_3d.npy')
            assert os.path.isfile(std_path), f"Could not find std_3d.npy in {std_path}."
            std_3d = np.load(std_path)

            self.__converter3d = CpuConverter3D(
                tflite_path=self._path_to_tflite_3d,
                mean_2d=mean_2d,
                std_2d=std_2d,
                mean_3d=mean_3d,
                std_3d=std_3d
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
        start_time = time.time()
        # 1. Preprocess image before feeding into the NN
        norm_img, new_h, new_w, original_in_size = self.__image_preprocessor(image)
        # 2. Feed the image into the NN and get PAF and heatmap tensors
        paf_pr, smoothed_heatmap_pr = self._model.predict(norm_img)
        # 3. Postprocess PAF and heatmap
        upsample_paf, indices, peaks = self._postprocess_np.process(heatmap=smoothed_heatmap_pr, paf=paf_pr)
        # 4. Build skeletons based off postprocessing results
        humans = SkeletBuilder.get_humans_by_PIF(peaks=peaks, indices=indices, paf_mat=upsample_paf)
        # 5. Scale skeletons' coordinates to the original image size
        scale_predicted_kp(
            predictions=[humans],
            model_size=(new_h, new_w),
            source_size=original_in_size
        )
        # 6. Perform additional correction
        updated_humans = modify_humans(humans)
        humans3d = None
        if self._path_to_tflite_3d is not None:
            humans_humans = [Human.from_array(x) for x in updated_humans]
            humans3d = self.__converter3d(humans_humans, image.shape[:-1])

        end_time = time.time() - start_time
        return PosePredictor.pack_data(humans=updated_humans, end_time=end_time, humans3d=humans3d)
