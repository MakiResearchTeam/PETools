import tensorflow as tf

from ..core import Model
from petools.tools.utils.tf_tools import load_graph_def


class CpuModelPB(Model):
    def __init__(
            self,
            pb_path, input_name,
            paf_name, heatmap_name,
            session=None
    ):
        self.__graph_def = load_graph_def(pb_path)
        self.__in_x = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='in_x')
        self.__upsample_size = tf.placeholder(dtype=tf.int32, shape=(2), name='upsample')
        self.__paf, self.__heatmap = tf.import_graph_def(
            self.__graph_def,
            input_map={
                input_name: self.__in_x
            },
            return_elements=[
                paf_name,
                heatmap_name
            ]
        )
        if session is None:
            session = tf.Session()
        self.__sess = session

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
        paf_pr, smoothed_heatmap_pr = self.__sess.run(
            [self.__paf, self.__heatmap],
            feed_dict={
                self.__in_x: norm_img
            }
        )

        return paf_pr, smoothed_heatmap_pr
