import tensorflow as tf


def load_graph_def(path):
    with tf.gfile.GFile(path, 'rb') as f:
        frozen_graph = tf.GraphDef()
        frozen_graph.ParseFromString(f.read())
    return frozen_graph
