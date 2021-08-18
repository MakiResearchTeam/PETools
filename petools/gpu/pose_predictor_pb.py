import json
import os
import time
import numpy as np
from typing import Callable
import tensorflow as tf

# Miscellaneous pose utilities
from petools.core import PosePredictorInterface
from petools.tools.estimate_tools.skelet_builder import SkeletBuilder
from petools.tools.utils import CAFFE, scale_predicted_kp
from petools.tools.utils.nns_tools.modify_skeleton import modify_humans
from petools.tools.estimate_tools import Human

from .utils import INPUT_TENSOR, IND_TENSOR, PAF_TENSOR, PEAKS_SCORE_TENSOR, INPUT_TENSOR_3D, OUTPUT_TENSOR_3D
from .gpu_model import GpuModel
from petools.model_tools.image_preprocessors import GpuImagePreprocessor
from petools.model_tools.operation_wrapper import OpWrapper, HumanModWrapper
from petools.model_tools.human_cleaner import HumanCleaner
from petools.model_tools.human_tracker import HumanTracker

# Init tools
from .init_utils import init_corrector, init_converter, init_smoother, init_classifier

OP_CONSTRUCTOR = Callable[[], object]
OP_INITIALIZER = Callable[..., OP_CONSTRUCTOR]


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
            path_to_pb_classifier: str = None,
            path_to_classifier_config: str = None,
            converter_initializer: OP_INITIALIZER = init_converter,
            corrector_initializer: OP_INITIALIZER = init_corrector,
            classifier_initializer: OP_INITIALIZER = init_classifier,
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
        path_to_pb_classifier : str
            Path to protobuf file with classification model.
        path_to_classifier_config : str
            Path to config for classification model.
        converter_initializer : Callable
            A function that takes in a path to protobuf and optional tf.Session object and returns
            a Callable[[], converter] that returns a converter when being called.
        corrector_initializer : Callable
            A function that takes in a path to protobuf and optional tf.Session object and returns
            a Callable[[], corrector] that returns a corrector when being called.
        classifier_initializer : Callable
            A function that takes in a path to protobuf, path to a file with a list of classes and optional tf.Session
            object and returns a Callable[[], classifier] that returns a classifier when being called.
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
        self.__path_to_pb_3d = path_to_pb_3d
        self.__path_to_pb_cor = path_to_pb_cor
        self.__path_to_config = path_to_config

        self.__path_to_pb_classifier = path_to_pb_classifier
        self.__path_to_classifier_config = path_to_classifier_config

        self.__converter_init = converter_initializer
        self.__corrector_init = corrector_initializer
        self.__classifier_init = classifier_initializer
        self._init_model()

    def _init_model(self):
        """
        Loads config and the estimate_tools's weights

        """
        with open(self.__path_to_config, 'r') as f:
            config = json.load(f)

        self.__model = GpuModel(
            pb_path=self.__path_to_tb,
            input_name=config[PosePredictor.INPUT_NAME],
            paf_name=config[PosePredictor.PAF_NAME],
            ind_name=config[PosePredictor.IND_TENSOR_NAME],
            peaks_score_name=config[PosePredictor.PEAKS_SCORE_NAME],
        )

        self.__session = self.__model.session

        self.__image_preprocessor = GpuImagePreprocessor(
            h=self.__min_h,
            w=int(self.__min_h / PosePredictor.W_BY_H),
            w_by_h=PosePredictor.W_BY_H,
            scale=PosePredictor.SCALE,
            norm_mode=self.__norm_mode
        )

        self.__skeleton_builder = SkeletBuilder()

        self.__human_cleaner = HumanCleaner(min_visible=8)
        # Will be initialized at first launch
        self.__tracker = None

        # --- SMOOTHER
        self.__smoother = HumanModWrapper(init_smoother())

        # --- CORRECTOR
        self.__corrector = lambda humans, **kwargs: humans
        if self.__path_to_pb_cor is not None:
            corrector_fn = self.__corrector_init(self.__path_to_pb_cor)
            self.__corrector = HumanModWrapper(corrector_fn)

        # --- CONVERTER
        self.__converter3d = lambda humans, **kwargs: humans
        if self.__path_to_pb_3d is not None:
            converter_fn = self.__converter_init(self.__path_to_pb_3d)
            self.__converter3d = HumanModWrapper(converter_fn)

        # --- CLASSIFIER
        self.__classifier = lambda humans, **kwargs: humans
        if self.__path_to_pb_classifier is not None:
            classifier_fn = self.__classifier_init(
                self.__path_to_pb_classifier,
                path_to_classifier_config=self.__path_to_classifier_config
            )
            self.__classifier = HumanModWrapper(classifier_fn)

    @property
    def pe_model(self):
        return self.__model

    @property
    def converter(self):
        return self.__converter3d

    @property
    def corrector(self):
        return self.__corrector

    @property
    def session(self):
        return self.__session

    @property
    def tracker(self):
        return self.__tracker

    def __human_tracker(self, humans, im_size):
        """
        Init tracker and keep update image size

        Parameters
        ----------
        humans: list
            List of class Human with predidction of NN
        im_size : tuple
            (Height, Wight) - of the image where prediction was taken

        """
        if self.__tracker is None:
            self.__tracker = HumanTracker(image_size=im_size)
        else:
            # If older im_size will be given - reset will be not applied
            # Otherwise all values will be dropped
            self.__tracker.reset(new_image_size=im_size)
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
                            (
                                human_id_1,
                                {   # 2D predictions
                                    'p0': [h1_x_1, h1_y_1, h1_v_1],
                                    'p1': [h1_x_2, h1_y_2, h1_v_2],
                                    ...
                                    'pn': [h1_x_n, h1_y_n, h1_v_n],
                                },

                                {
                                    # 3D predictions
                                    'p0': [h1_x_1, h1_y_1, h1_z_1, h1_v_1],
                                    'p1': [h1_x_2, h1_y_2, h1_z_2, h1_v_2],
                                    ...
                                    'pn': [h1_x_n, h1_y_n, h1_z_n, h1_v_n],
                                },
                            ),
                            ...
                            ...

                            (
                                human_id_N,
                                {
                                    'p0': [hN_x_1, hN_y_1, hN_v_1],
                                    'p1': [hN_x_2, hN_y_2, hN_v_2],
                                    ...
                                    'pn': [hN_x_n, hN_y_n, hN_v_n],
                                },

                                {
                                    'p0': [hN_x_1, hN_y_1, hN_z_1, hN_v_1],
                                    'p1': [hN_x_2, hN_y_2, hN_z_2, hN_v_2],
                                    ...
                                    'pn': [hN_x_n, hN_y_n, hN_z_n, hN_v_n],
                                },
                            ),
                    ],
                    PosePredictor.TIME: some_float_number
                }
            Where PosePredictor.HUMANS and PosePredictor.TIME - are strings ('humans' and 'time')
        """
        # Summary of overall pipeline:
        # 1. Preprocess image;
        # 2. Predict with NN;
        # 3. Get skeletons by prediction from NN;
        # 4. Scale prediction on original image;
        # 5. Modify skeletons in production style;
        # 6. Clean humans (aka predictions), delete skeletons with low number of keypoints;
        # 7. Track humans, assign unique id for every prediction and track human further;
        # 8. Smoother, smooth predictions;
        # 9. Corrector, correct predictions;
        # 10. Converter 3d, convert 2d predictions into 3d.
        # 11. Classify 2d points.
        # At the end - pack results (2d, 3d and time of overall pipeline) into dict

        # Measure time of overall pipeline
        start_time = time.time()
        # 1. Process image, norm_img feeds into NN
        # new_h, new_w - size of the image `norm_img`
        # origin_in_size - size of the original image
        norm_img, new_h, new_w, original_in_size = self.__image_preprocessor(image)
        # 2. Take prediction
        batched_paf, indices, peaks = self.__model.predict(norm_img)
        # 3. Get skeletons by prediction from NN
        # Keep input size into model fresh for builder
        self.__skeleton_builder.set_img_size((new_h, new_w))
        # Take humans (skeletons)
        humans = self.__skeleton_builder.get_humans_by_PIF(peaks=peaks, indices=indices, paf_mat=batched_paf[0])
        # 4. Scale prediction to original image
        scale_predicted_kp(
            predictions=[humans],
            model_size=(new_h, new_w),
            source_size=original_in_size
        )
        # 5. Modify skeletons in production style
        # Transform points from training format to the inference one.
        # Returns a numpy of shape [n_humans, n_points, 3]
        humans = modify_humans(humans)
        # Transfer numpy array into Human class
        humans = [Human.from_array(x) for x in humans]
        # 6. Clean humans (aka predictions), delete skeletons with low number of keypoints
        # Remove skeletons with low number of keypoints
        humans = self.__human_cleaner(humans)
        # 7. Track humans, assign unique id for every prediction and track human further;
        humans = self.__human_tracker(humans, original_in_size)
        # 8. One Euro algorithm which smooth keypoints movement
        humans = self.__smoother(humans)
        # 9. Corrector, correct predictions;
        # Corrector need source resolution to perform human normalization
        humans = self.__corrector(humans, source_resolution=original_in_size)
        # 10. Converter 3d, convert 2d predictions into 3d.
        # Converter need source resolution to perform human normalization
        humans = self.__converter3d(humans, source_resolution=original_in_size)
        # 11. Classify 2d points.
        humans = self.__classifier(humans)

        # Time of the overall prediction pipeline
        end_time = time.time() - start_time
        # Pack data into suitable for other APIs form
        return PosePredictor.pack_data(humans=humans, end_time=end_time)

    def predict_debug(self, image: np.ndarray):
        """
        Estimate poses on single image
        !NOTICE
        This methods only for debug purposes. it can produce different result compare to `predict` method,
        In order to take predictions, call `predict`

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
                            (
                                human_id_1,
                                {   # 2D predictions
                                    'p0': [h1_x_1, h1_y_1, h1_v_1],
                                    'p1': [h1_x_2, h1_y_2, h1_v_2],
                                    ...
                                    'pn': [h1_x_n, h1_y_n, h1_v_n],
                                },

                                {
                                    # 3D predictions
                                    'p0': [h1_x_1, h1_y_1, h1_z_1, h1_v_1],
                                    'p1': [h1_x_2, h1_y_2, h1_z_2, h1_v_2],
                                    ...
                                    'pn': [h1_x_n, h1_y_n, h1_z_n, h1_v_n],
                                },
                            ),
                            ...
                            ...

                            (
                                human_id_N,
                                {
                                    'p0': [hN_x_1, hN_y_1, hN_v_1],
                                    'p1': [hN_x_2, hN_y_2, hN_v_2],
                                    ...
                                    'pn': [hN_x_n, hN_y_n, hN_v_n],
                                },

                                {
                                    'p0': [hN_x_1, hN_y_1, hN_z_1, hN_v_1],
                                    'p1': [hN_x_2, hN_y_2, hN_z_2, hN_v_2],
                                    ...
                                    'pn': [hN_x_n, hN_y_n, hN_z_n, hN_v_n],
                                },
                            ),
                    ],
                    PosePredictor.TIME: some_float_number
                }
            Where PosePredictor.HUMANS and PosePredictor.TIME - are strings ('humans' and 'time')
        """
        # Summary of overall pipeline:
        # 1. Preprocess image;
        # 2. Predict with NN;
        # 3. Get skeletons by prediction from NN;
        # 4. Scale prediction on original image;
        # 5. Modify skeletons in production style;
        # 6. Clean humans (aka predictions), delete skeletons with low number of keypoints;
        # 7. Track humans, assign unique id for every prediction and track human further;
        # 8. Smoother, smooth predictions;
        # 9. Corrector, correct predictions;
        # 10. Converter 3d, convert 2d predictions into 3d.
        # 11. Classify 2d points.
        # At the end - pack results (2d, 3d and time of overall pipeline) into dict

        # Measure time of overall pipeline
        start_time = time.time()
        start_time_preprocess = time.time()
        # 1. Process image, norm_img feeds into NN
        # new_h, new_w - size of the image `norm_img`
        # origin_in_size - size of the original image
        norm_img, new_h, new_w, original_in_size = self.__image_preprocessor(image)
        end_time_preprocess = time.time() - start_time_preprocess

        start_time_predict = time.time()
        # 2. Take prediction
        batched_paf, indices, peaks = self.__model.predict(norm_img)
        end_time_predict = time.time() - start_time_predict

        start_time_pafprocess = time.time()
        # 3. Get skeletons by prediction from NN
        # Keep input size into model fresh for builder
        self.__skeleton_builder.set_img_size((new_h, new_w))
        # Take humans (skeletons)
        humans = self.__skeleton_builder.get_humans_by_PIF(peaks=peaks, indices=indices, paf_mat=batched_paf[0])
        end_time_pafprocess = time.time() - start_time_pafprocess

        start_time_scale_pred = time.time()
        # 4. Scale prediction to original image
        scale_predicted_kp(
            predictions=[humans],
            model_size=(new_h, new_w),
            source_size=original_in_size
        )
        end_time_scale_pred = time.time() - start_time_scale_pred

        start_time_modify = time.time()
        # 5. Modify skeletons in production style
        # Transform points from training format to the inference one.
        # Returns a numpy of shape [n_humans, n_points, 3]
        humans = modify_humans(humans)
        # Transfer numpy array into Human class
        humans = [Human.from_array(x) for x in humans]
        end_time_modify = time.time() - start_time_modify

        start_time_cleaner = time.time()
        # 6. Clean humans (aka predictions), delete skeletons with low number of keypoints
        # Remove skeletons with low number of keypoints
        humans = self.__human_cleaner(humans)
        end_time_cleaner = time.time() - start_time_cleaner

        start_time_treacker = time.time()
        # 7. Track humans, assign unique id for every prediction and track human further;
        humans = self.__human_tracker(humans, original_in_size)
        end_time_tracker = time.time() - start_time_treacker

        start_time_euro = time.time()
        # 8. One Euro algorithm which smooth keypoints movement
        humans = self.__smoother(humans)
        end_time_euro = time.time() - start_time_euro

        start_time_corrector = time.time()
        # 9. Corrector, correct predictions;
        # Corrector need source resolution to perform human normalization
        humans = self.__corrector(humans, source_resolution=original_in_size)
        end_time_corrector = time.time() - start_time_corrector

        start_time_converter = time.time()
        # 10. Converter 3d, convert 2d predictions into 3d.
        # Converter need source resolution to perform human normalization
        humans = self.__converter3d(humans, source_resolution=original_in_size)
        end_time_converter = time.time() - start_time_converter

        start_time_classifier = time.time()
        # 11. Classify 2d points.
        humans = self.__classifier(humans)
        end_time_classifier = time.time() - start_time_classifier

        # Time of the overall prediction pipeline
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
            'converter3d': end_time_converter,
            'classifier': end_time_classifier
        }
        # Pack data into suitable for other APIs form
        return PosePredictor.pack_data(humans=humans, end_time=end_time, **data_time_logs)


if __name__ == '__main__':
    print(PosePredictor.HUMANS)

