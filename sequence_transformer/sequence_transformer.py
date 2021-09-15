import tensorflow as tf
from sequence_transformer.constants import RESERVED_TOKENS
from sequence_transformer.transformer import Transformer
from sequence_transformer.constants import INPUT_PAD, CLASSIFICATION_TOKEN, SEPARATOR_TOKEN
from sequence_transformer.training_utils import load_vocabulary


class TransformerInputPrep:
    """
    This class formats the input sequence(s) to a Transformer by concatenating them and inserting appropriate tokens.
    The resulting sequence will have the following format:

    [CLS] [SEP] seq_1 [SEP] seq_2 [SEP] seq_3 [SEP] ...

    This is inspired by how BERT formats its input. The CLS token can be used (as BERT does) as a summary of the entire
    input. For example, binary classification for sentiment analysis in NLP, or purchase intention in click-stream.

    The SEP token is used to separates each sequence from the next (or from the CLS token).
    """
    def __init__(self, seq_chain_mapping):
        """
        Args:
            seq_chain_mapping: Specifies how to existing features should be chained to form new ones.

        Example:
        Let's say the feature set includes the following tensors:
        session_items, basket_items, sessions_events, basket_events

        We can create two new features, items and events, by chaining the existing ones as follows:

        seq_chain_mapping = {
                             'items': ['session_items', 'basket_items'],
                             'events': ['sessions_events', 'basket_events']
                            }
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

    def __call__(self, features, keep_features=False):
        """
        Args:
            features: Dictionary of feature, mapping feature names to tensors
            keep_features: Whether to keep chained features in the returned feature dictionary

        Returns:
            A features dictionary containing the newly created features. It will leave untouched all other features that
            were not involved in chaining.
        """
        features = features.copy()  # traced functions cannot change their input
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

        # 3 - Removing old features used in creating the new ones (if so specified by keep_features argument)
        if not keep_features:  # Drop features that were used in forming sequence pairs
            features_to_drop = set()
            for seq_pair in self.seq_chain_mapping.values():
                features_to_drop = features_to_drop.union(set(seq_pair))
            features = {k: v for k, v in features.items() if k not in features_to_drop}

        return features, segment_starts, segment_ends


class SequenceTransformer(tf.keras.models.Model):

    """
    This model closely resembles that of Chen et. al. (2019), available at https://arxiv.org/pdf/1905.06874.pdf.

    It consists, for the most part, of a Transformer which takes as input a series of sequences
    (e.g. sequence of items viewed). This Transformer then creates contextualized embeddings that correspond one for one
    to input items. These embeddings are then passed through a BinaryClassificationHead (which can be a series of fully-connected layers)
    to produce the output. Depending on the task, one can decide what type of BinaryClassificationHead to use and which embeddings to pass
    through it. This is similar to how various classification layers can be mounted on top of a BERT model to perform
    different tasks (binary classification, sequence tagging, multi-class classification with a softmax, etc.)

    For instance, one can only send the output corresponding to CLS through the binary classification BinaryClassificationHead. The CLS
    token can be trained to "summarize" the entire chain of input sequences. Or in the case of predicting returns given
    the click-stream, one could feed the click-stream + basket (two sequences) and then pass the output of the
    Transformer that corresponds to basket items to the BinaryClassificationHead and get a Tensor of the same length as basket.
    The model can then be trained so that it predicts the probability of return for each item in the basket. (one would
    need to set a maximum basket size, as the output of the model has to have a fixed length, and then pad when the
    basket is smaller than that).

    Another example is scoring items for recommendation, given the click-stream up to a certain point. In this case,
    sequence 1 of input would be click-stream and sequence 2 will always be of length 1, containing a single target item
    to be scored. One can then feed the output corresponding to the target item through the BinaryClassificationHead and produce a relevance
    score.

    The following is an example of how to specify the configuration of inputs:

    ```python

    sequential_input_config = {
        'items': ['seq_1_items', 'seq_2_items'],
        'events': ['seq_1_events', 'seq_2_events']
    }

    feature_vocabs = {
        'items': '../data/vocabs/item_vocab.txt',
        'events': '../data/vocabs/event_vocab.txt'
    }

    embedding_dims = {
        'items': 20,
        'events': 4
    }
    ```
    In the above example the input contains two sequences. These, for example, could be session and basket in the task
    of predicting returns. In addition, each sequence consists of two (separately embedded) elements. Items, and their
    corresponding events, line (view, handbag) or (add_to_basket, shoe). These components will be concatenated before
    going into the transformer. The final (batched) result will have the shape

    (batch_size, seq_1_len + seq_2_len, d_model)

    where d_model is the sum of the embedding dimension for each components (items and event in the above example).
    """

    def __init__(self,
                 sequential_input_config,
                 feature_vocabs,
                 embedding_dims,
                 head_unit,
                 segment_to_head=None,  # ToDo: Find a better name for this. Also, it should allow multiple segments
                 value_to_head=None,  # Find a better name, like token_to_output
                 num_encoder_layers=1,
                 num_attention_heads=1,
                 dropout_rate=0.1,
                 **kwargs):
        """

        Args:
            sequential_input_config: Configuration of the sequential part of input, which is the part that will be
                fed to the Transformer.
            feature_vocabs: Dictionary mapping categorical feature names to vocabulary files.
            embedding_dims: Dictionary that specifies the embedding dimension for embedded features.
            head_unit: The Head (e.g. dense layers) that will consume the output of the Transformer and produce the final output.
            segment_to_head: Which segment/sequence (0-based, where 0 is always the CLS token) to feed to the BinaryClassificationHead
                of the mode. This parameter needs a better name.
            value_to_head: The outputs corresponding to input positions that have this value will be sent to the BinaryClassificationHead
                Example: value_to_head = '[Mask]' will send Transformer output for '[Mask]' tokens in the input sequence
                to the BinaryClassificationHead
            num_encoder_layers: Number of Encoder layers
            num_attention_heads: Number of Attention heads in each Encoder layer
            dropout_rate: Dropout rate
            final_layers_dims: Dimensions of fully connected layers that come after Transformer and produce output. This
                determines the BinaryClassificationHead of the model. In the future this will be replaced by an argument or arguments that
                specify what type of BinaryClassificationHead should be used.
        """
        super(SequenceTransformer, self).__init__(**kwargs)

        self.sequential_input_config = sequential_input_config
        self.feature_vocabs = feature_vocabs
        self.embedding_dims = embedding_dims
        self.head = head_unit
        self.num_encoder_layers = num_encoder_layers
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate  # ToDo: Do all of these need to be stored as class attributes?

        assert (segment_to_head is not None or value_to_head is not None) and \
               (segment_to_head is None or value_to_head is None), \
               "Exactly one of segment_to_head and value_to_head must be provided."

        self.segment_to_head = segment_to_head
        self.value_to_head = value_to_head

        self.transformer_input_prep = TransformerInputPrep(self.sequential_input_config)
        # self.token_mapper = TokenMapper(vocabularies=feature_vocabs, reserved_tokens=RESERVED_TOKENS)
        self.vocab_lookup_tables = self._create_lookup_tables(vocabularies=self.feature_vocabs,
                                                              tokens_to_prepend=RESERVED_TOKENS)
        # This operation will fail with KeyError if there is a key in embedding_dims that is not present in
        # feature_vocabs. In other words, if you imply that a feature must be embedded (by specifying an embedding
        # dimension) for it, but don't provide a vocab file for it.
        # Note: The .numpy() below converts the tf.Tensor object to numpy. Otherwise, it will throw a 'Not serializable'
        # error when you try to serialize and save the model
        self.embedding_sizes = {f: self.vocab_lookup_tables[f].size().numpy() for f in self.feature_vocabs.keys()}

        # Transformer
        self.transformer = Transformer(
            embedding_sizes=self.embedding_sizes,
            embedding_dims=self.embedding_dims,
            num_layers=self.num_encoder_layers,
            num_attention_heads=self.num_attention_heads,
            encoder_ff_dim=100,
            dropout_rate=self.dropout_rate
        )

    def get_config(self):
        """
        Custom Keras layers and models are not serializable unless they override this method.
        """
        config = super(SequenceTransformer, self).get_config()
        config.update({
            'sequential_input_config': self.sequential_input_config,
            'feature_vocabs': self.feature_vocabs,
            'embedding_dims': self.embedding_dims,
            'head_unit': self.head_unit,
            'segment_to_head': self.segment_to_head,
            'value_to_head': self.value_to_head,
            'num_encoder_layers': self.num_encoder_layers,
            'num_attention_heads': self.num_attention_heads,
            'dropout_rate': self.dropout_rate
        })
        return config

    @staticmethod
    def _create_lookup_tables(vocabularies, tokens_to_prepend=None):
        lookup_tables = {}
        for feature_name, vocab_file in vocabularies.items():
            keys = load_vocabulary(vocab_file)  # words in vocab file
            if tokens_to_prepend is not None:
                keys = tf.concat([tokens_to_prepend, keys], axis=0)  # prepend reserved tokes to the vocabulary
            values = tf.convert_to_tensor(range(len(keys)), dtype=tf.int64)
            initializer = tf.lookup.KeyValueTensorInitializer(keys=keys, values=values)
            lookup_tables[feature_name] = tf.lookup.StaticVocabularyTable(initializer, num_oov_buckets=1)

        return lookup_tables

    @staticmethod
    def _gather_output_by_raw_value(transformer_output, key_raw_feature, filter_value):
        """
            Given a raw feature (not embedded) of shape (batch_size, seq_len), it first finds positions of
            `value_to_head` values. It then gathers the embeddings that correspond to these entries from transformer's
            output. The catch is, different examples may have different number of entries that match this value.
            So we need to do a ragged gather_nd and then pad the resulting ragged tensor to make it rectangular.

        Args:
            transformer_output:
            key_raw_feature: See comments about *the ugly patch* in the call method
            filter_value: What value to look for. This is the value specified in value_to_head argument, e.g. [MASK]

        Returns:
            Transformed embeddings from Transformer's output that correspond to the entries that matched `filter_value`
            Example: if `filter_value` = [MASK] the outputs corresponding to [MASK] positions in the input will be
                gathered and returned.
        """
        batch_size = tf.shape(key_raw_feature)[0]

        # (num_of_matching_tokens, 2). This loses the batch dimension and puts all the indices on the same axis
        indices = tf.where(tf.equal(key_raw_feature, filter_value))

        # Since all examples in the batch don't necessarily have the same number of matching entries, we will create
        # a RaggedTensor of indices. The resulting gathered tensor will be padded later.
        ragged_row_ids = indices[:, 0]  # The index of examples in the batch (0 for 1st example, 1 for 2nd, etc.)

        # Specifying nrows ensures batch size is preserved, even if some examples have 0 matching tokens
        ragged_indices = tf.RaggedTensor.from_value_rowids(values=indices, value_rowids=ragged_row_ids,
                                                           nrows=tf.cast(batch_size, tf.int64))

        ragged_matching_embeddings = tf.gather_nd(params=transformer_output, indices=ragged_indices)

        # The indices we used for gather_nd were ragged and so is the result. We need to pad and make it rectangular
        # Pad parts will later be ignored in the loss function because the label is (must be) padded accordingly.
        matching_embeddings = ragged_matching_embeddings.to_tensor(default_value=INPUT_PAD)

        return matching_embeddings

    @tf.function
    def call(self, inputs, training=None, mask=None):

        # Traced functions are not allowed to change their input arguments
        raw_features, segment_starts, segment_ends = self.transformer_input_prep(features=inputs)

        # We might need raw_features later. Also, we don't want to throw away feature that don't need vocab mapping
        features = raw_features.copy()
        for feature_name, lookup_table in self.vocab_lookup_tables.items():
            features[feature_name] = lookup_table.lookup(features[feature_name])

        # separating sequential features that should go into the Transformer from potentially non-seq "side" features
        sequential_features = {feature_name: features[feature_name]
                               for feature_name in self.sequential_input_config.keys()}

        # ToDo: Side features should be added to the output of the Transformer
        transformer_output = self.transformer(sequential_features, training, mask)  # (batch_size, seq_len, d_model)

        if self.segment_to_head is not None:
            # segment_starts and segment_ends mark the index in the Transformer's output tensor where each sequence
            # begins and ends. The first segment corresponds to the [CLS] token and is therefore always
            # (batch_size, 1, d_model)
            # The rest are (batch_size, seq_i_len, d_model) for the i-th sequence.
            head_input = transformer_output[:, segment_starts[self.segment_to_head]:segment_ends[self.segment_to_head], ...]
        elif self.value_to_head is not None:
            # # # # # # This is an ugly patch
            # We designed the Transformer so that it would accept multiple sequences and stack them. Most obvious
            # example is a sequence of items and a sequence of events associated with those items, e.g.
            #   (view, bag), (add_to_cart, shoe), ...
            # This is not common in the literature at all.
            # Here, we are trying to send some of the elements from the output of the Transformer to the
            # BinaryClassificationHead, and we're doing so based on input values. For instance, only send positions that
            # correspond to [MASK] inputs. But it is not clear which input sequence we should be looking at?
            # The items? or the events? Given how rare this multi-feature input is, I think this is an unnecessary
            # complication. And the Transformer should only expect one input feature: The items.
            some_raw_feature_name = list(self.sequential_input_config.keys())[0]
            # (batch_size, seq_len)
            some_raw_feature = raw_features[some_raw_feature_name]
            # # # # # # end of ugly patch
            head_input = self._gather_output_by_raw_value(transformer_output, some_raw_feature, self.value_to_head)

        else:
            raise ValueError("One of value_to_head and segment_to_head must be provided.")

        logits = self.head(head_input)

        # In serving, the model can accept and return an instance id
        if 'instance_id' in inputs.keys():
            return {
                'instance_id': inputs['instance_id'],
                'logits': logits
            }
        else:
            return logits

    def get_serving_signature(self):
        # This can be improved in a couple of ways.
        # This does not allow non-sequential side features. It also assumes all sequential features are string
        # The latter assumption is also made in the model implementation where we require dictionaries for seq. inputs
        # So that we can map them to integers. But what if the sequential input is already an integer?

        seq_feature_list = []
        for seq_feature_chain in self.sequential_input_config.values():
            # this loop compiles a list of all features that were chained together to form sequential inputs
            # Example:
            # sequential_input_config = {'items': ['session_items', 'basket_items']}
            # But when calling the model the signature will be
            # {'session_items': ..., 'basket_items': ...}
            seq_feature_list.extend(seq_feature_chain)

        instance_id_dict = {'instance_id': tf.TensorSpec([], dtype=tf.string)}
        features_dict = {
            seq_feature: tf.TensorSpec([None, None], dtype=tf.string)
            for seq_feature in seq_feature_list
        }

        return features_dict  # {**instance_id_dict, **features_dict}

    # # ToDo: write a helper function that produces these given the arguments to init
    # @tf.function(input_signature=[
    #     tf.TensorSpec([None, None], dtype=tf.string)
    # ])
    # def model_server(self, **kwargs):
    #     return {'scores': self.call(inputs=kwargs)}
