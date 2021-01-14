import tensorflow as tf


class BinaryClassificationHead(tf.keras.layers.Layer):

    def __init__(self, dense_layer_dims, **kwargs):

        super(BinaryClassificationHead, self).__init__(kwargs)

        self.intermediate_layers = [tf.keras.layers.Dense(layer_dim, activation='relu') for layer_dim in dense_layer_dims]
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs, **kwargs):

        # items go through these layers without interacting with each other. So pads don't affect real data
        x = inputs
        for dense_layer in self.intermediate_layers:
            # (batch_size, input_len, d_model)
            x = dense_layer(x)  # A DNN layer acts on the last axis only. So items (and pads) won't interact

        logits = self.output_layer(x)  # logits.shape == (batch_size, input_len, 1)

        # (batch_size, seq_len)
        logits = tf.squeeze(logits, axis=-1)

        return logits


class SoftMaxHead(tf.keras.layers.Layer):

    def __init__(self, dense_layer_dims, output_vocab_size, **kwargs):

        super(SoftMaxHead, self).__init__(kwargs)

        self.dense_layer_dims = dense_layer_dims
        self.output_vocab_size = output_vocab_size

        self.intermediate_layers = [tf.keras.layers.Dense(layer_dim, activation='relu')
                                    for layer_dim in self.dense_layer_dims]
        self.output_layer = tf.keras.layers.Dense(units=self.output_vocab_size, activation=tf.keras.activations.softmax)

    def get_config(self):
        """
        Custom Keras layers and models are not serializable unless they override this method.
        """
        config = super(SoftMaxHead, self).get_config()
        config.update({
            'dense_layer_dims': self.dense_layer_dims,
            'output_vocab_size': self.output_vocab_size
        })
        return config

    def call(self, inputs, **kwargs):

        # items go through these layers without interacting with each other. So pads don't affect real data
        x = inputs  # (batch_size, input_len, d_model)
        for dense_layer in self.intermediate_layers:
            x = dense_layer(x)  # A DNN layer acts on the last axis only. So items (and pads) won't interact

        logits = self.output_layer(x)  # logits.shape == (batch_size, input_len, output_layer_dim)

        return logits
