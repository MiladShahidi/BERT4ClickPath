import tensorflow as tf


def load_vocabulary(vocab_file):
    with tf.io.gfile.GFile(vocab_file, 'r') as f:
        return tf.strings.strip(f.readlines())
