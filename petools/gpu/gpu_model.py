import tensorflow.compat.v1 as tf

from ..core import Model, ProtobufModel


class GpuModel(ProtobufModel, Model):
    def __init__(
            self,
            pb_path, input_name,
            paf_name, ind_name, peaks_score_name, upsample_size_name='upsample_size',
            session=None
    ):
        super(GpuModel, self).__init__(
            protobuf_path=pb_path,
            input_map={
                input_name: ('float32', [1, None, None, 3], 'in_x'),
                upsample_size_name: ('int32', [2], 'upsample')
            },
            output_tensors=[paf_name, ind_name, peaks_score_name],
            session=session
        )

        self._in_x = self.input_map[input_name]
        self._upsample_size = self.input_map[upsample_size_name]

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
                self._in_x: norm_img,
                self._upsample_size: resize_to
        })
        return batched_paf, indices, peaks
