import tensorflow.compat.v1 as tf

from ..core import Model, ProtobufModel


class GpuModel(ProtobufModel, Model):
    def __init__(
            self,
            pb_path, input_name,
            paf_name, ind_name, peaks_score_name, upsample_size_name='upsample_size',
            session=None
    ):
        # Create graph for GPU model
        graph = tf.Graph()
        with graph.as_default():
            # Create some placeholders inside created graph
            self.__in_x = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='in_x')
            self.__upsample_size = tf.placeholder(dtype=tf.int32, shape=(2), name='upsample')

        super(GpuModel, self).__init__(
            protobuf_path=pb_path,
            input_map={
                input_name: self.__in_x,
                upsample_size_name: self.__upsample_size
            },
            output_tensors=[paf_name, ind_name, peaks_score_name],
            graph=graph,
            session=session
        )

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
        batched_paf, indices, peaks = super(GpuModel, self).predict(feed_dict={
                self.__in_x: norm_img,
                self.__upsample_size: resize_to
        })
        return batched_paf, indices, peaks
