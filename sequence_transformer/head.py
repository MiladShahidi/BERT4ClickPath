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

        self.intermediate_layers = [tf.keras.layers.Dense(layer_dim, activation='relu') for layer_dim in dense_layer_dims]
        self.output_layer = tf.keras.layers.Dense(units=output_vocab_size, activation=tf.keras.activations.softmax)

    def call(self, inputs, **kwargs):

        # items go through these layers without interacting with each other. So pads don't affect real data
        x = inputs  # (batch_size, input_len, d_model)
        for dense_layer in self.intermediate_layers:
            x = dense_layer(x)  # A DNN layer acts on the last axis only. So items (and pads) won't interact

        logits = self.output_layer(x)  # logits.shape == (batch_size, input_len, output_layer_dim)

        return logits


class MultiLabel_MultiClass_classification(tf.keras.layers.Layer):

    def __init__(self, dense_layer_dims, output_vocab_size, **kwargs):

        super(MultiLabel_MultiClass_classification, self).__init__(kwargs)

        self.intermediate_layers = [tf.keras.layers.Dense(layer_dim, activation='relu') for layer_dim in dense_layer_dims]
        self.output_layer = tf.keras.layers.Dense(units=output_vocab_size, activation='sigmoid')

    def call(self, inputs, **kwargs):
        # items go through these layers without interacting with each other. So pads don't affect real data
        x = inputs  # (batch_size, input_len, d_model)
        for dense_layer in self.intermediate_layers:
            x = dense_layer(x)  # A DNN layer acts on the last axis only. So items (and pads) won't interact

        logits = self.output_layer(x)  # logits.shape == (batch_size, input_len, output_layer_dim)

        logits = tf.squeeze(logits, axis=1)

        return logits
