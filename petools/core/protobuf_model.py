import tensorflow.compat.v1 as tf

from petools.tools.utils.tf_tools import load_graph_def


class ProtobufModel:

    def __init__(
            self, protobuf_path: str, input_map: dict,
            output_tensors: list, graph: tf.Graph,
            session: tf.Session = None):
        """
        A utility for using models stored in a protobuf file.

        Parameters
        ----------
        protobuf_path : str
            Path to the protobuf file.
        input_map : dict
            Input mapping of graph tensors to their placeholders.
            Example: { 'input_tensor_name': tf.placeholder }
        output_tensors : list
            List of names of the output tensors. Example: [ 'output_tensor_name:0' ]. NOTE that there is
            a zero appended to the tensor name.
        graph : tf.Graph
            Graph which contains of operations/variables that must be executed by this class
            For more info, refer to official docs of tensorflow 1.15

        """
        assert len(input_map) != 0
        assert len(output_tensors) != 0
        self._input_map = input_map
        self.__protobuf_graph = graph

        with self.__protobuf_graph.as_default():
            # Load frozen graph into `graph`
            self._graph_def = load_graph_def(protobuf_path)
            self._output_tensors = tf.import_graph_def(
                self._graph_def,
                input_map=input_map,
                return_elements=output_tensors
            )
            if session is None:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                session = tf.Session(config=config)
            self._sess = session

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

        # Execute session with graph
        with self.__protobuf_graph.as_default():
            return self._sess.run(
                self._output_tensors,
                feed_dict=feed_dict
            )
