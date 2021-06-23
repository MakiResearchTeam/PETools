import tensorflow as tf

from petools.tools.utils.tf_tools import load_graph_def


def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


class ProtobufModel:
    def __init__(self, protobuf_path: str, input_tensor_names: list, output_tensors_names: list):
        """
        A utility for using models stored in a protobuf file.

        Parameters
        ----------
        protobuf_path : str
            Path to the protobuf file.
        input_tensor_names : list
            Input mapping of graph tensors to their placeholders.
            Example: { 'input_tensor_name': tf.placeholder }
        output_tensors_names : list
            List of names of the output tensors. Example: [ 'output_tensor_name:0' ]. NOTE that there is
            a zero appended to the tensor name.
        """
        assert len(input_tensor_names) != 0
        assert len(output_tensors_names) != 0
        self._graph_def = load_graph_def(protobuf_path)
        self._input_map = input_tensor_names
        self._model_fn = wrap_frozen_graph(
            self._graph_def,
            inputs=input_tensor_names,
            outputs=output_tensors_names
        )

    def predict(self, *data):
        """
        Runs the graph from the protobuf.

        Parameters
        ----------
        feed_dict : dict
            Contains pairs ('input_tensor_name', data_tensor) or (tf.placeholder, data_tensor)

        Returns
        -------
        list
            Output tensors' values.
        """
        tensor_data = []
        for x in data:
            tensor_data.append(tf.convert_to_tensor(x))
        return [x.numpy() for x in self._model_fn(*tensor_data)]

