import tensorflow as tf

from ..core import Model
from petools.tools.utils.tf_tools import load_graph_def


class GpuModel(Model):
    def __init__(self, pb_path, input_name, paf_name, ind_name, peaks_score_name, upsample_size_name='upsample_size'):
        self.__graph_def = load_graph_def(pb_path)
        self.__in_x = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='in_x')
        self.__upsample_size = tf.placeholder(dtype=tf.int32, shape=(2), name='upsample')
        self.__resized_paf, self.__indices, self.__peaks_score = tf.import_graph_def(
            self.__graph_def,
            input_map={
                input_name: self.__in_x,
                upsample_size_name: self.__upsample_size
            },
            return_elements=[
                paf_name,
                ind_name,
                peaks_score_name
            ]
        )
        self.__sess = tf.Session()

    def predict(self, norm_img):
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

        return batched_paf, indices, peaks
