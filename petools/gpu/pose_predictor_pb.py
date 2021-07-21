import json
import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
import pathlib

# Miscellaneous pose utilities
from petools.core import PosePredictorInterface
from petools.tools.estimate_tools.skelet_builder import SkeletBuilder
from petools.tools.utils import CAFFE, scale_predicted_kp
from petools.tools.utils.nns_tools.modify_skeleton import modify_humans
from petools.tools.estimate_tools import Human

from .utils import INPUT_TENSOR, IND_TENSOR, PAF_TENSOR, PEAKS_SCORE_TENSOR, INPUT_TENSOR_3D, OUTPUT_TENSOR_3D
from .gpu_model import GpuModel
from petools.model_tools.image_preprocessors import GpuImagePreprocessor
from petools.model_tools.operation_wrapper import OPWrapper
from petools.model_tools.human_cleaner import HumanCleaner
from petools.model_tools.human_tracker import HumanTracker
from petools.model_tools.one_euro_filter import OneEuroModule
# Converter / Corrector
from petools.model_tools.transformers import HumanProcessor, Transformer, PoseTransformer
from petools.model_tools.transformers import Postprocess3D, Postprocess2D, Preprocess3D, Preprocess2D, SequenceBuffer
from petools.model_tools.transformers.utils import H36_2DPOINTS_DIM_FLAT


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
            path_to_pb_cor: str = None,
            min_h=320,
            expected_w=600,
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
        path_to_pb_3d : str
            Path to protobuf file with 2d-3d converter model.
        path_to_pb_cor : str
            Path to protobuf file with corrector model.
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
        self.__expected_w = expected_w
        self.__norm_mode = norm_mode
        self.__path_to_tb = path_to_pb
        self.__path_to_tb_3d = path_to_pb_3d
        self.__path_to_tb_cor = path_to_pb_cor
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

        self.__human_cleaner = HumanCleaner(min_visible=5)
        # Will be initialized at first launch
        self.__tracker = None

        # --- SMOOTHER
        self.__smoother = OPWrapper(lambda: OneEuroModule())

        human_processor = HumanProcessor.init_from_lib()
        # --- CORRECTOR
        self.__corrector = lambda humans, **kwargs: humans
        if self.__path_to_tb_cor is not None:
            corrector_t = Transformer(protobuf_path=self.__path_to_tb_cor, session=self.__sess)
            corrector_fn = lambda: PoseTransformer(
                transformer=corrector_t,
                seq_buffer=SequenceBuffer(dim=H36_2DPOINTS_DIM_FLAT, seqlen=32),
                preprocess=Preprocess2D(human_processor),
                postprocess=Postprocess2D(human_processor)
            )
            self.__corrector = OPWrapper(corrector_fn)

        # --- CONVERTER
        self.__converter3d = lambda humans, **kwargs: humans
        if self.__path_to_tb_3d is not None:
            converter_t = Transformer(protobuf_path=self.__path_to_tb_3d, session=self.__sess)
            converter_fn = lambda: PoseTransformer(
                transformer=converter_t,
                seq_buffer=SequenceBuffer(dim=H36_2DPOINTS_DIM_FLAT, seqlen=32),
                preprocess=Preprocess3D(human_processor),
                postprocess=Postprocess3D(human_processor)
            )
            self.__converter3d = OPWrapper(converter_fn)

    def __human_tracker(self, humans, im_size):
        if self.__tracker is None:
            self.__tracker = HumanTracker(image_size=im_size)
        return self.__tracker(humans)

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

        start_time_preprocess = time.time()
        norm_img, new_h, new_w, original_in_size = self.__image_preprocessor(image)
        end_time_preprocess = time.time() - start_time_preprocess

        start_time_predict = time.time()
        batched_paf, indices, peaks = self.__model.predict(norm_img)
        end_time_predict = time.time() - start_time_predict

        start_time_pafprocess = time.time()
        humans = SkeletBuilder.get_humans_by_PIF(peaks=peaks, indices=indices, paf_mat=batched_paf[0])
        humans = [human_s.to_np() for human_s in humans]
        end_time_pafprocess = time.time() - start_time_pafprocess

        start_time_scale_pred = time.time()
        # Scale prediction to original image
        scale_predicted_kp(
            predictions=[humans],
            model_size=[new_h, new_w],
            source_size=original_in_size
        )
        end_time_scale_pred = time.time() - start_time_scale_pred

        # Transform points from training format to the inference one. Returns a list of shape [n_humans, n_points, 3]
        start_time_modify = time.time()
        # humans = [human_s.to_np() for human_s in humans]
        humans = modify_humans(humans)
        end_time_modify = time.time() - start_time_modify

        start_time_cleaner = time.time()
        humans = self.__human_cleaner(humans)
        end_time_cleaner = time.time() - start_time_cleaner

        start_time_treacker = time.time()
        humans = [Human.from_array(x) for x in humans]
        humans = self.__human_tracker(humans, image.shape[:-1])
        end_time_tracker = time.time() - start_time_treacker

        # One Euro algorithm for smoothing keypoints movement
        start_time_euro = time.time()
        humans = self.__smoother(humans)
        end_time_euro = time.time() - start_time_euro

        # Corrector need source resolution to perform human normalization
        start_time_corrector = time.time()
        humans = self.__corrector(humans, source_resolution=image.shape[:-1])
        end_time_corrector = time.time() - start_time_corrector

        # Converter need source resolution to perform human normalization
        start_time_converter = time.time()
        humans = self.__converter3d(humans, source_resolution=image.shape[:-1])
        end_time_converter = time.time() - start_time_converter

        end_time = time.time() - start_time

        data_time_logs = {
            'preprocess': end_time_preprocess,
            'predict': end_time_predict,
            'pafprocess': end_time_pafprocess,
            'scale kp': end_time_scale_pred,
            'modify': end_time_modify,
            'clean': end_time_cleaner,
            'tracker': end_time_tracker,
            'euro': end_time_euro,
            'corrector': end_time_corrector,
            'converter3d': end_time_converter
        }

        return PosePredictor.pack_data(humans=humans, end_time=end_time, **data_time_logs)


if __name__ == '__main__':
    print(PosePredictor.HUMANS)

