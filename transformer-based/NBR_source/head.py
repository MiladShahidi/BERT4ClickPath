import tensorflow as tf


class Head(tf.keras.layers.Layer):

    def __init__(self, dnn_layer_dims, **kwargs):

        super(Head, self).__init__(kwargs)

        self.intermediate_layers = [tf.keras.layers.Dense(layer_dim, activation='relu') for layer_dim in dnn_layer_dims]
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs, **kwargs):

        # items go through these layers without interacting with each other. So pads don't affect real data
        x = inputs
        for dnn_layer in self.intermediate_layers:
            # (batch_size, seq_len, d_model)
            x = dnn_layer(x)  # A DNN layer acts on the last axis only. So items (and pads) won't interact

        logits = self.output_layer(x)  # logits.shape == (batch_size, seq_len, output_layer_dim)

        # (batch_size, seq_len)
        logits = tf.squeeze(logits, axis=-1)

        return logits
