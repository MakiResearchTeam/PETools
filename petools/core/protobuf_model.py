import tensorflow.compat.v1 as tf
from typing import Dict, Tuple, Union

from petools.tools.utils.tf_tools import load_graph_def

DTYPE = Union[str, tf.DType]
SHAPE = Union[list, tuple]
NAME = str
PLACEHOLDER_INFO = Tuple[DTYPE, SHAPE, NAME]


class ProtobufModel:

    def __init__(
            self, protobuf_path: str, input_map: Dict[NAME, PLACEHOLDER_INFO],
            output_tensors: list,
            session: tf.Session = None):
        """
        A utility for using models stored in a protobuf file.

        Parameters
        ----------
        protobuf_path : str
            Path to the protobuf file.
        input_map : dict
            Input mapping of graph tensors to their placeholders.
            Example: { input_tensor_name: (dtype, shape, name) }
        output_tensors : list
            List of names of the output tensors. Example: [ 'output_tensor_name:0' ]. NOTE that there is
            a zero appended to the tensor name.
        """
        assert len(input_map) != 0
        assert len(output_tensors) != 0
        self._session = session

        if self._session is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self._session = tf.Session()

        # This graph management is required for usage of TensorRT.
        # TensorRT can't run when there are multiple graphs in one session.
        # Therefore, we need an individual session for each model running on TensorRT.
        with self._session.graph.as_default():
            # Parse input_map into tf.placeholders
            self._input_map = {}
            for tensor_name, (dtype, shape, name) in input_map.items():
                self._input_map[tensor_name] = tf.placeholder(dtype=dtype, shape=shape, name=name)

            # Load frozen graph into `graph`
            self._graph_def = load_graph_def(protobuf_path)
            self._output_tensors = tf.import_graph_def(
                self._graph_def,
                input_map=input_map,
                return_elements=output_tensors
            )

    @property
    def input_map(self):
        return self._input_map

    def predict(self, feed_dict):
        """
        Execute the graph from the protobuf.

        Parameters
        ----------
        feed_dict : dict
            Contains pairs ('input_tensor_name', data_tensor) or (tf.placeholder, data_tensor)

        Returns
        -------
        list
            Output tensors' values.

        """
        for input_map in feed_dict.keys():
            if isinstance(input_map, str):
                assert input_map in self._input_map.keys(), f'Unknown tensor name: {input_map}. Expected one of those:' \
                                                            f'{self._input_map.keys()}'
            else:
                assert input_map in self._input_map.values(), f'Unknown tensor: {input_map}. Expected one of those:' \
                                                            f'{self._input_map.values()}'

        return self._session.run(
            self._output_tensors,
            feed_dict=feed_dict
        )
