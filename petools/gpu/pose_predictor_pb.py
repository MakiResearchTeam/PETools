import json
import os
import time
import numpy as np
import tensorflow as tf
import pathlib

from petools.core import PosePredictorInterface
from petools.tools.estimate_tools.skelet_builder import SkeletBuilder
from petools.tools.utils import CAFFE, scale_predicted_kp
from petools.tools.utils.nns_tools.modify_skeleton import modify_humans
from petools.tools.estimate_tools import Human

from .utils import INPUT_TENSOR, IND_TENSOR, PAF_TENSOR, PEAKS_SCORE_TENSOR, INPUT_TENSOR_3D, OUTPUT_TENSOR_3D
from .gpu_model import GpuModel
from ..image_preprocessors import GpuImagePreprocessor
from .gpu_converter_3d import GpuConverter3D


class PosePredictor(PosePredictorInterface):
    INPUT_NAME = INPUT_TENSOR
    IND_TENSOR_NAME = IND_TENSOR
    PAF_NAME = PAF_TENSOR
    PEAKS_SCORE_NAME = PEAKS_SCORE_TENSOR
    UPSAMPLE_SIZE = 'upsample_size'
    W_BY_H = 1128.0 / 1920.0

    def __init__(
            self,
            path_to_pb: str,
            path_to_config: str,
            path_to_pb_3d: str = None,
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
        self.__path_to_tb_3d = path_to_pb_3d
        self.__path_to_config = path_to_config
        self._init_model()

    def _init_model(self):
        """
        Loads config and the estimate_tools's weights

        """
        with open(self.__path_to_config, 'r') as f:
            config = json.load(f)

        self.__sess = tf.Session()

        self.__model = GpuModel(
            pb_path=self.__path_to_tb,
            input_name=config[PosePredictor.INPUT_NAME],
            paf_name=config[PosePredictor.PAF_NAME],
            ind_name=config[PosePredictor.IND_TENSOR_NAME],
            peaks_score_name=config[PosePredictor.PEAKS_SCORE_NAME],
            session=self.__sess
        )

        self.__image_preprocessor = GpuImagePreprocessor(
            h=self.__min_h,
            w=int(self.__min_h / PosePredictor.W_BY_H),
            w_by_h=PosePredictor.W_BY_H,
            scale=PosePredictor.SCALE,
            norm_mode=self.__norm_mode
        )

        if self.__path_to_tb_3d is not None:
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

            self.__converter3d = GpuConverter3D(
                pb_path=self.__path_to_tb_3d,
                mean_2d=mean_2d,
                std_2d=std_2d,
                mean_3d=mean_3d,
                std_3d=std_3d,
                input_name=config[INPUT_TENSOR_3D],
                output_name=config[OUTPUT_TENSOR_3D],
                session=self.__sess
            )

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

        # Measure time of prediction
        start_time = time.time()
        norm_img, new_h, new_w, original_in_size = self.__image_preprocessor(image)
        batched_paf, indices, peaks = self.__model.predict(norm_img)
        humans = SkeletBuilder.get_humans_by_PIF(peaks=peaks, indices=indices, paf_mat=batched_paf[0])
        # Scale prediction to original image
        scale_predicted_kp(
            predictions=[humans],
            model_size=(new_h, new_w),
            source_size=image.shape[:-1]
        )
        # Transform points from training format to the inference one. Returns a list of shape [n_humans, n_points, 3]
        updated_humans = modify_humans(humans)
        humans_humans = [Human.from_array(x) for x in updated_humans]
        humans3d = None
        if self.__path_to_tb_3d is not None:
            humans3d = self.__converter3d(humans_humans, image.shape[:-1])

        end_time = time.time() - start_time
        return PosePredictor.pack_data(humans=updated_humans, end_time=end_time, humans3d=humans3d)


if __name__ == '__main__':
    print(PosePredictor.HUMANS)

