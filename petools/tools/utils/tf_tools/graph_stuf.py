import tensorflow as tf


def load_graph_def(path):
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(open(path, 'rb').read())
    return graph_def
