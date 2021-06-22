from ..core import Model
import tensorflow.compat.v1 as tf


class CpuModel(Model):
    def __init__(self, tflite_file, num_threads):
        interpreter = tf.lite.Interpreter(model_path=str(tflite_file), num_threads=num_threads)
        interpreter.allocate_tensors()
        self.__interpreter = interpreter
        self.__in_x = interpreter.get_input_details()[0]["index"]

        self.__paf_tensor = interpreter.get_output_details()[0]["index"]
        self.__heatmap_tensor = interpreter.get_output_details()[1]["index"]
        # TODO: In most our models for CPU, placeholder upsample_size is not used
        # TODO: But it can be used in further updates
        # TODO: Think how fix this, for now - leave it as it is
        # self.__upsample_size = interpreter.get_input_details()[1]["index"]

    def predict(self, norm_img):
        interpreter = self.__interpreter
        interpreter.set_tensor(self.__in_x, norm_img)
        # TODO: In most our models for CPU, placeholder upsample_size is not used
        # TODO: But it can be used in further updates
        # TODO: Think how fix this, for now - leave it as it is
        # interpreter.set_tensor(self.__upsample_size, self.__resize_to)
        # Run estimate_tools
        interpreter.invoke()

        paf_pr, smoothed_heatmap_pr = (
            interpreter.get_tensor(self.__paf_tensor),
            interpreter.get_tensor(self.__heatmap_tensor)
        )
        return paf_pr, smoothed_heatmap_pr
