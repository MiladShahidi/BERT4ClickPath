import tensorflow as tf
from constants import RESERVED_TOKENS
from transformer import Transformer
from constants import UNKNOWN_INPUT, MISSING_EVENT_OR_ITEM
from constants import CLS, INPUT_PAD, SEP, CLASSIFICATION_TOKEN, SEPARATOR_TOKEN
from head import Head


class TransformerInputPrep:

    def __init__(self, seq_chain_mapping):
        """
        Example:

        ```python
        seq_pair_mapping = {
                            'items': ['session_items', 'basket_items'],
                            'events': ['sessions_events', 'basket_events']
                        }
        ```

        """
        self.seq_chain_mapping = seq_chain_mapping

    @staticmethod
    def _chain_sequences(sequences):
        """
        Args:
            sequences: A list containing (1 or more) sequence tensors to be chained

        Returns:
            The tensor resulting from concatenating the two input sequences and adding appropriate tokens
            output looks like: [CLS] seq_1 [SEP] seq_2 [SEP] seq_3 [SEP] seq_4 ...
        """
        session_dims = tf.unstack(tf.shape(sequences[0]))  # This is a python list
        session_dims[1] = 1  # token is concatenated along axis 1. So it should have size 1 on this axis
        token_shape = session_dims
        cls_token = tf.fill(dims=token_shape, value=tf.cast(CLASSIFICATION_TOKEN, sequences[0].dtype))
        sep_token = tf.fill(dims=token_shape, value=tf.cast(SEPARATOR_TOKEN, sequences[0].dtype))

        sequences = [cls_token] + sequences

        # insert sep in between sequences. Result will be [cls] [sep] seq_1 [sep] seq_2 [sep] ...
        # The first sep is inserted for ease of calculating segment start and end positions. BERT doesn't have that sep.
        concat_list = [sequences[i // 2] if i % 2 == 0 else sep_token for i in range(2 * len(sequences))]

        # Notice axis=1. This doesn't care whether or not there are more dimensions after 1
        seq_chain = tf.concat(concat_list, axis=1)

        return seq_chain

    def __call__(self, features, replace=True):
        """
        Args:
            seq_1: (batch_size, seq_1_len)
            seq_2: (batch_size, seq_2_len)

        Returns:

        """
        # 1 - Chaining features as specified
        for new_feature, seq_pair in self.seq_chain_mapping.items():
            sequence_pair_list = [features[name] for name in seq_pair]
            features[new_feature] = self._chain_sequences(sequence_pair_list)

        # 2 - Calculating start and end positions of sequences

        # Using the SEP token to find out where each sequence begins and ends (including the leading CLS)
        # Sequences in the same position are supposed to have the same length. So doesn't matter which one we use.
        # I'll just use whichever chained feature that happens to be first
        some_chained_feature = features[list(self.seq_chain_mapping.keys())[0]]
        # Similarly, the lengths of sequences are the same along the batch dimension. So, I'll get rid of batch axis
        sample_chained_tensor = some_chained_feature[0, :]  # taking first (or any) sample in the batch
        # The following is (n, d), where d is # dimensions in condition of where. in this case 1 (see tf.where docs)
        sep_positions = tf.where(tf.equal(sample_chained_tensor, SEPARATOR_TOKEN))
        # So we can simply squeeze it out and get a 1-d tensor of indexes
        segment_ends = tf.squeeze(sep_positions, axis=1)  # 1-d tensor containing positions of SEP tokens
        # Each sequence starts right after the previous one's SEP token. First one starts at 0.
        segment_starts = tf.concat([[0], segment_ends[:-1]+1], axis=0)  # Making sure the first start pos is 0

        # 3 - Removing old features used in creating the new ones (if so specified by replace argument)
        if replace:  # Drop features that were used in forming sequence pairs
            features_to_drop = set()
            for seq_pair in self.seq_chain_mapping.values():
                features_to_drop = features_to_drop.union(set(seq_pair))
            features = {k: v for k, v in features.items() if k not in features_to_drop}

        return features, segment_starts, segment_ends


class TokenMapper:
    # ToDo: Find a better name for this class
    """
    This class maps tokens (words) to their index in the vocabulary, taking into account the fact that we need
    to add a list of reserved tokens to the vocabulary before mapping.

    inputs
    dict mapping variable name to vocab file for embeddings. item and event vocabs are mandatory.
        could include more for other vars that must be embedded
    """
    def __init__(self, vocabularies, reserved_tokens, **kwargs):

        # In some cases, multiple variables might share the same vocab file. But it's not efficient to create multiple
        # copies of the same lookup table for them. So we will do this in 2 steps to avoid duplicate lookup tables
        lookup_per_vocab_file = {
            vocab_file: self._create_lookup_table(vocab_file, tokens_to_prepend=reserved_tokens)
            for vocab_file in set(vocabularies.values())
        }

        self.lookup_tables = {var_name: lookup_per_vocab_file[vocab_file]
                              for var_name, vocab_file in vocabularies.items()}

    def _create_lookup_table(self, vocab_file, tokens_to_prepend):
        keys = self.load_vocabulary(vocab_file)  # words in vocab file
        keys = tf.concat([tokens_to_prepend, keys], axis=0)  # prepend reserved tokes to the vocabulary
        values = tf.convert_to_tensor(range(len(keys)), dtype=tf.int64)
        initializer = tf.lookup.KeyValueTensorInitializer(keys=keys, values=values)
        return tf.lookup.StaticVocabularyTable(initializer, num_oov_buckets=1)

    @staticmethod
    def load_vocabulary(vocab_file):
        with tf.io.gfile.GFile(vocab_file, 'r') as f:
            return tf.strings.strip(f.readlines())

    def _apply_vocabs(self, features):
        result = features.copy()  # Traced functions (by TF for exporting) should not change their input arguments
        # features for which a vocab file was not specified won't have a lookup table and will go through unchanged.
        for var_name in self.lookup_tables.keys():
            result[var_name] = self.lookup_tables[var_name].lookup(result[var_name])

        return result

    def __call__(self, features):
        return self._apply_vocabs(features)


class ReturnsModel(tf.keras.models.Model):

    def __init__(self,
                 input_seq_mapping,  # Sequences will be formatted according to this before being fed to Transformer
                 feature_vocabs,
                 embedding_dims,
                 num_encoder_layers,
                 num_attention_heads,
                 dropout_rate,
                 final_layers_dims,
                 **kwargs):

        super(ReturnsModel, self).__init__(**kwargs)

        self.embedding_dims = embedding_dims
        # self.item_vocab_size = len(self._load_vocabulary(self.item_vocab_file))
        # self.event_vocab_size = len(self._load_vocabulary(self.event_vocab_file))
        self.num_encoder_layers = num_encoder_layers
        self.encoder_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate  # ToDo: Do all of these need to be stored as to class attributes?

        self.transformer_input_prep = TransformerInputPrep(input_seq_mapping)
        self.token_mapper = TokenMapper(vocabularies=feature_vocabs, reserved_tokens=RESERVED_TOKENS)

        # Verify that embedding_dim and feature_vocabs dicts have the same keys
        assert set(feature_vocabs.keys()) == set(embedding_dims.keys()),\
            "The feature_vocabs and embedding_dims dictionaries must have the same set of keys."

        embedding_sizes = {f: self.token_mapper.lookup_tables[f].size() for f in feature_vocabs.keys()}

        # Transformer
        self.transformer = Transformer(
            embedding_sizes=embedding_sizes,
            embedding_dims=embedding_dims,
            num_layers=self.num_encoder_layers,
            encoder_attention_heads=self.encoder_attention_heads,
            encoder_ff_dim=100,
            maximum_position_encoding=10000,
            dropout_rate=self.dropout_rate
        )

        self.head = Head(dnn_layer_dims=final_layers_dims)

    def format_features(self, features):
        # session_len and basket_size are those of longest example in the batch

        # (batch_size, session_len)
        session_events = self._fillna(base_tensor=features['event_name'],
                                      fill_tensor=features['page_type'],
                                      na_value=MISSING_EVENT_OR_ITEM)

        # (batch_size, session_len)
        session_items = self._fillna(base_tensor=features['product_skn_id_events'],
                                     fill_tensor=features['product_skn_id_page_views'],
                                     na_value=MISSING_EVENT_OR_ITEM)

        # (batch_size, basket_size)
        basket_events = tf.fill(dims=tf.shape(features['basket_product_id']),
                                value=tf.cast(BUY_EVENT, session_events.dtype))

        # add shippping charge
        per_item_feature_names = [  # (batch_size, basket_size)
            'ordered_quantity',
            # 'shipping_charge',
            'unit_price',
            'discount'
        ]

        per_item_feature_list = [features[k] for k in per_item_feature_names]
        # per_user_feature_list = [features[k] for k in per_user_feature_names]
        all_side_features = per_item_feature_list  # + per_user_feature_list

        # (batch_size, basket_size, n_per_item_features)
        basket_side_features = tf.stack(all_side_features, axis=-1)

        # (batch_size, session_len)
        session_shaped_padding = tf.fill(dims=tf.shape(session_events), value=INPUT_PAD)
        # (batch_size, session_len, n_per_item_features)
        session_shaped_padding = tf.stack(values=[session_shaped_padding] * len(all_side_features), axis=-1)
        session_shaped_padding = tf.cast(session_shaped_padding, basket_side_features.dtype)

        items = self._format_seq_pair(session_items, features['basket_product_id'])
        events = self._format_seq_pair(session_events, basket_events)
        side_features = self._format_seq_pair(session_shaped_padding, basket_side_features)

        return items, events, side_features

    def call(self, inputs, training=None, mask=None):

        # Traced functions are not allowed to change their input arguments
        # features = format_transformer_input(inputs, seq_pair_mapping=self.input_seq_mapping)
        features, segment_starts, segment_ends = self.transformer_input_prep(features=inputs)
        features = self.token_mapper(features)

        x = self.transformer(features, training, mask)  # (batch_size, seq_len, d_model)

        # segment_starts and segment_ends marks the index in the Transformer's output tensor where each sequence begins
        # and ends. The first segment corresponds to the [CLS] token and is therefore always
        # (batch_size, 1, d_model)
        # The rest are (batch_size, seq_i_len, d_model) for the i-th sequence.

        # This model feeds the output corresponding to the second sequence. This is specific to this particular task
        head_input = x[:, segment_starts[2]:segment_ends[2], ...]

        logits = self.head(head_input)

        return logits

    # ToDo: write a helper function that produces these given the arguments to init
    # @tf.function(input_signature=[
    #     # Basket features: (batch_size, basket_size)
    #     tf.TensorSpec([None, None], dtype=tf.string),   # basket_product_id
    #     tf.TensorSpec([None, None], dtype=tf.float32),  # discount
    #     tf.TensorSpec([None, None], dtype=tf.float32),  # ordered_quantity
    #     tf.TensorSpec([None, None], dtype=tf.float32),  # unit_price
    #     # Order features: (batch_size, 1)
    #     tf.TensorSpec([None, 1], dtype=tf.float32),  # shipping_charge
    #     # Session: (batch_size, session_length)
    #     tf.TensorSpec([None, None], dtype=tf.string),  # event_name
    #     tf.TensorSpec([None, None], dtype=tf.string),  # page_type
    #     tf.TensorSpec([None, None], dtype=tf.string),  # product_skn_id_events
    #     tf.TensorSpec([None, None], dtype=tf.string),  # product_skn_id_page_views
    # ])
    # def model_server(self,
    #                  # Basket features
    #                  basket_product_id,
    #                  discount,
    #                  ordered_quantity,
    #                  unit_price,
    #                  # Order features
    #                  shipping_charge,
    #                  # Session
    #                  event_name,
    #                  page_type,
    #                  product_skn_id_events,
    #                  product_skn_id_page_views):
    #
    #     features = {
    #         # Basket features
    #         'basket_product_id': basket_product_id,
    #         'discount': discount,
    #         'ordered_quantity': ordered_quantity,
    #         'unit_price': unit_price,
    #         # Order features
    #         'shipping_charge': shipping_charge,
    #         # Session
    #         'event_name': event_name,
    #         'page_type': page_type,
    #         'product_skn_id_events': product_skn_id_events,
    #         'product_skn_id_page_views': product_skn_id_page_views
    #     }
    #
    #     return {'scores': self.call(inputs=features, training=False)}
