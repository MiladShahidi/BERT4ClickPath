import tensorflow as tf
from sequence_transformer.constants import INPUT_PAD, SEP, ITEM_EMBEDDING_LAYER_NAME
import numpy as np


def create_segment_markers(seq, sep=SEP):
    """
    Example:
        seq =  # (batch of 2), SEP is 4 in the vocab
            [
                [  3   4   1 444   1 903 186   1 947   1 798   0   0   0   0   0   0   4 814 706 959 537   4]
                [  3   4 169   1 714 169 999 696 737 320  11 666 493 229 859   1  77   4 662 990   0   0   4]
            ]

        result:
            [
                [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 3]
                [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 3]
            ]

    Notice that because each sequence is padded before being concatenated, SEP (4) appears in the same position in all
        examples across the batch. However, this function does not make that assumption and will work fine regardless.

    Args:
        seq: 2-D tensor with shape (batch_size, seq_len)
        sep: The marker separating the segments (SEP token)

    Returns:
        a 2-D tensor of the same shape as seq, containing the sequence number of corresponding elements in seq
    """
    tf.debugging.assert_rank(seq, 2, 'Expected 2-D tensor')
    sep_pos = tf.cast(tf.where(tf.equal(seq, sep), 1, 0), tf.int32)
    segment_markers = tf.cumsum(sep_pos, axis=1)
    return segment_markers


# ToDo: Keras now provides native padding and masking layers. Consider using those instead of manual masking
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, INPUT_PAD), tf.float32)
    # add extra dimensions so it will match attention logits. We need to add the padding to the attention logits later.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Note: Look ahead mask is not used in our case

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        # This saves all arguments as attributes of the object so that they can ve saved when the model is exported
        # This same mapping should be added to the layer's/model's config. See get_config
        self.num_heads = num_heads
        self.d_model = d_model

        assert self.d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(self.d_model)
        self.wk = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)

        self.dense = tf.keras.layers.Dense(self.d_model)

    def get_config(self):
        """
        Custom Keras layers and models are not serializable unless they override this method.
        """
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads
        })
        return config

    # TODO: Why do they split q, k and v between heads? The paper doesn't do this.
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='gelu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        # This saves all arguments as attributes of the object so that they can be saved when the model is exported
        # This same mapping should be added to the layer's/model's config. See get_config
        self.dff = dff
        self.rate = rate
        self.d_model = d_model
        self.num_heads = num_heads

        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)

    def get_config(self):
        """
        Custom Keras layers and models are not serializable unless they override this method.
        """
        config = super(EncoderLayer, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate
        })
        return config

    def call(self, x, training=None, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)

        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 dropout_rate, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        # This saves all arguments as attributes of the object so that they can be saved when the model is exported
        # This same mapping should be added to the layer's config. See get_config
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        # self.maximum_position_encoding = maximum_position_encoding

        # TODO: It is possible to have the embedding layer produce a mask tensor by setting mask_zero=True
        #  In that case all subsequent layers must support masking.
        #  See: https://www.tensorflow.org/guide/keras/masking_and_padding
        #  This could be a more robust and fool-proof way of implementing masking. But before spending time on this
        #  consider the fact that this will make everything more dependant on the keras embedding layer and will make it
        #  harder to remove it. We might want to replace this with an embedding feature_column, in case the latter
        #  provides more flexibility, e.g. loading embeddings from checkpoints or manipulating embeddings manually.

        self.enc_layers = [EncoderLayer(self.d_model, self.num_heads, self.dff, self.dropout_rate)
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def get_config(self):
        """
        Custom Keras layers and models are not serializable unless they override this method.
        """
        config = super(Encoder, self).get_config()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config

    def call(self, inputs, training=None, mask=None):
        # seq_len = tf.shape(inputs)[1]
        # inputs = self.embedding(inputs)  # (batch_size, input_seq_len, d_model)

        # inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # inputs += self.pos_encoding[:, :seq_len, :]

        inputs = self.dropout(inputs, training=training)

        for i in range(self.num_layers):
            inputs = self.enc_layers[i](inputs, training, mask)

        return inputs  # (batch_size, input_seq_len, d_model)


class Transformer(tf.keras.layers.Layer):
    """
    An encoder-only Transformer. See "Attention Is All You Need", Vaswani, et. al. (2017).

    https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

    Note that this is a Keras Layer, not Model. It's meant to be used as part of (i.e. a layer inside) other models.

    It receives a dictionary of input features (see the call method) each of which must have the same shape of:

    (batch_size, seq_len)

    It will then turn these into embeddings (of possibly different dimensions) and concatenate them along the
    embedding axis. So, each one becomes (batch_size, seq_len, dim_i) and the concatenated result will be:

    (batch_size, seq_len, d_model) where d_model = sum of dim_i for all embedding dimensions

    """
    # This is the maximum number of input sequences (as marked and separated by the SEP token) this model can accept
    # The only reason we need to specify this, is to be able to create a segment embedding layer in advance.
    # If segment embeddings are not used this becomes irrelevant
    MAX_INPUT_SEQ = 100

    def __init__(self,
                 num_layers,
                 num_attention_heads,
                 embedding_sizes,
                 embedding_dims,
                 encoder_ff_dim,
                 dropout_rate,
                 item_embedding_weights=None,
                 **kwargs):

        """

        Args:
            embedding_sizes: Size of the vocabulary for each input feature. One can think of this as the number of
                rows in each embedding lookup table.
            embedding_dims: Dimension of embedding vector for input features. One can think of this as the number of
                columns in each embedding lookup table.
            num_layers: Number of Encoder layers.
            encoder_attention_heads: Number of Attention heads.
            encoder_ff_dim: Dimension of the feed forward layers inside the Encoder (see "Attention is All You Need").
            dropout_rate: Dropout rate
            item_embedding_weights: Not currently implemented, but can be used to load pre-trained embeddings from a
                checkpoint. See the instantiation of the Embedding layers.
            **kwargs:
        """

        super(Transformer, self).__init__(**kwargs)

        assert set(embedding_sizes.keys()) == set(embedding_dims.keys()), \
            "embedding_sizes and embedding_dims must have the same set of keys."

        # Some of these attributes are not used as such, but assigning them to a class attribute allows TF to trace them
        # This is needed for serialization. I'm not completely sure though.
        # See https://www.tensorflow.org/guide/keras/custom_layers_and_models
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.embedding_sizes = embedding_sizes
        self.embedding_dims = embedding_dims
        self.encoder_ff_dim = encoder_ff_dim
        self.dropout_rate = dropout_rate
        self.item_embedding_weights = item_embedding_weights

        self.maximum_position_encoding = 10000  # This used to be an argument. But doesn't need to be.

        self.d_model = sum(embedding_dims.values())

        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=self.d_model,
            num_heads=num_attention_heads,
            dff=encoder_ff_dim,
            dropout_rate=dropout_rate
        )

        self.embedding_layers = {
            feature: tf.keras.layers.Embedding(
                input_dim=embedding_sizes[feature],
                output_dim=embedding_dims[feature],
                # The followings are to enable loading from checkpoints
                # weights=embedding_weights,  # I verified that passing None to this is equivalent to not passing it
                # name=EMBEDDING_LAYER_NAME  # Ensures that we know the name of this layer and can later load by name
            )
            for feature in embedding_dims.keys()
        }

        self.pos_encoding = positional_encoding(self.maximum_position_encoding, self.d_model)
        # self.segment_embedding_layer = tf.keras.layers.Embedding(self.MAX_INPUT_SEQ, self.d_model)

    def get_config(self):
        """
        Custom Keras layers and models are not serializable unless they override this method.
        """
        config = super(Transformer, self).get_config()
        config.update({
            'num_layers': self.num_layers,
            'num_attention_heads': self.num_attention_heads,
            'embedding_sizes': self.embedding_sizes,
            'embedding_dims': self.embedding_dims,
            'encoder_ff_dim': self.encoder_ff_dim,
            'dropout_rate': self.dropout_rate,
            'item_embedding_weights': self.item_embedding_weights
        })
        return config

    def call(self, inputs, training=None, mask=None):
        some_feature = inputs[list(inputs.keys())[0]]
        seq_len = tf.shape(some_feature)[1]  # features are (batch_size, seq_len) before being embedded

        # All sequence features are padded the same way. It doesn't matter which one is used to create the mask
        padding_mask = create_padding_mask(some_feature)  # (batch_size, seq_len)

        # Convert to embeddings
        embedded_features = {feature_name: self.embedding_layers[feature_name](inputs[feature_name])
                             for feature_name in inputs.keys()}

        # Concatenate embedded_features along their last dimension
        embedded_seq = tf.concat(list(embedded_features.values()), axis=-1)  # (batch_size, seq_len, d_model)
        # d_model == sum(embedding_dims.values())
        embedded_seq *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Segment embeddings
        # segment_markers = create_segment_markers(some_feature)  # (batch_size, seq_len)
        # segment_embeddings = self.segment_embedding_layer(segment_markers)  # (batch_size, seq_len, d_model)
        # embedded_seq += segment_embeddings  # (batch_size, seq_len, d_model)

        # ToDo: Try learnable positional embeddings from Bert4Rec: https://arxiv.org/pdf/1904.06690.pdf
        embedded_seq += self.pos_encoding[:, :seq_len, :]

        encoder_output = self.encoder(inputs=embedded_seq, training=training, mask=padding_mask)  # (batch_size, seq_len, d_model)

        return encoder_output


if __name__ == '__main__':
    pass
