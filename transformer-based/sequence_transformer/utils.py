import tensorflow as tf


def load_vocabulary(vocab_file):
    if tf.io.gfile.isdir(vocab_file):
        # Strangely enough GFile does not raise an error when it is given a directory to read from.
        # Reported this on Github: https://github.com/tensorflow/tensorflow/issues/46282#issue-782000566
        raise IsADirectoryError(f'{vocab_file} is a directory.')

    with tf.io.gfile.GFile(vocab_file, 'r') as f:
        return tf.strings.strip(f.readlines())
