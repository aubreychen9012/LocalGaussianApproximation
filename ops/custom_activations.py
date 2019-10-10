import tensorflow as tf


def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
