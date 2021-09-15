import tensorflow as tf
from clickstream_transformer.constants import INPUT_PADDING_TOKEN, LABEL_PAD, INPUT_PAD, INPUT_MASKING_TOKEN
from source.data_generator import ClickStreamGenerator
from clickstream_transformer.clickstream_transformer import ClickstreamTransformer
from source.cloze_constants import MAX_MASKED_ITEMS, MASKED_PERCENTAGE, modes
from clickstream_transformer.head import SoftMaxHead
from clickstream_transformer.training_utils import load_vocabulary
import functools
import os


def parse_examples(serialized_example, feature_spec):
    features = tf.io.parse_example(serialized=serialized_example, features=feature_spec)

    # A VarLenFeature is always parsed to a SparseTensor
    for key in features.keys():
        if isinstance(features[key], tf.SparseTensor):
            features[key] = tf.sparse.to_dense(features[key])

    return features


def random_choice(x, size, axis=0, preserve_order=True):
    """
    Returns randomly chosen *indices* from serialized_string.
    This is the similar to np.random.choice. But it always returns unique choices, i.e. keep_features=False in numpy
    """
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices)[:size]
    if preserve_order:
        sample_index = tf.sort(sample_index)

    return sample_index


def format_labels(feature_dict):
    """
        This mapping is applied after batching. It adds padding (-1) to the left of labels (which are associated with
        seq_2 items). The seq_1 component does not have labels and should be assigned pads instead of labels.
    """
    # This was written for the specific case of two part inputs for HBC. Needs to be updated
    raise NotImplementedError
    batch_size = tf.shape(feature_dict['event_name'])[0]
    session_len = tf.shape(feature_dict['event_name'])[1]

    session_and_tokens_label_padding = tf.fill(
        dims=(batch_size, session_len+2),  # +2 allows for the two tokens (CLS and SEP) that will be in the input tensor
        value=LABEL_PAD
    )
    left_padded_labels = tf.concat([
        session_and_tokens_label_padding,
        feature_dict['labels']
    ], axis=1)

    feature_dict['labels'] = left_padded_labels

    return feature_dict


def random_item_mask(item_list, masked_percentage, max_masked):
    """
    Args:
        item_list: A tensor containing a list of item IDs (strings).

    Returns:
        Randomly masks some of the items and returns the masked item list along with the list of masked items, which can
            be used to create the labels tensor.
    """
    session_len = tf.cast(tf.size(item_list), tf.float32)
    n_masked = tf.cast(tf.multiply(session_len, masked_percentage), dtype=tf.int32)
    n_masked = tf.clip_by_value(n_masked, clip_value_min=0, clip_value_max=max_masked)
    # Randomly pick n_masked items to mask from the list of items
    mask_index = random_choice(item_list, n_masked)
    masked_item_list, masked_items = mask_items(item_list, mask_index)
    return masked_item_list, masked_items


def mask_items(item_list, mask_index):
    masked_items = tf.gather(item_list, mask_index, axis=0)
    # Mask the chosen items by replacing them with the masking token
    mask = tf.fill(dims=tf.shape(mask_index), value=INPUT_MASKING_TOKEN)
    # The reshaping that is done for indices below, used to be applied to the other two inputs as well.
    # This was probably done to make this function work with higher dimension tensors (e.g. batched tensors)
    # It is not required when this function is applied as a mapping to individual 1-D tensors. So I removed it.
    masked_item_list = tf.tensor_scatter_nd_update(
        tensor=item_list,
        indices=tf.reshape(mask_index, (-1, 1)),
        updates=mask,
    )

    return masked_item_list, masked_items


def cloze_data_prep(features, mode, label_lookup_table):

    def label_map(x):
        """ Labels cannot be mapped later. We need to turn then into integers here """
        return tf.cast(label_lookup_table.lookup(x), tf.float32)

    if mode == modes.TRAIN:  # This is from the outer scope
        # last item is saved for validation. Should be removed when training
        for seq_feature in ['asin', 'unixReviewTime']:
            features[seq_feature] = features[seq_feature][:-1]

        special_example = False
        # special_example = (random_choice([0, 1, 2], 1) == 0)  # Random number generation is tricky inside a map fn
        if special_example:
            last_item_index = tf.expand_dims(tf.size(features['asin']) - 1, axis=0)  # Needs to be 1-D tensor
            features['asin'], labels = mask_items(features['asin'], last_item_index)
        else:
            features['asin'], labels = random_item_mask(
                item_list=features['asin'],
                masked_percentage=MASKED_PERCENTAGE,
                max_masked=MAX_MASKED_ITEMS
            )
    elif mode == modes.EVAL:  # validation data
        if False:
            # Masks only the last item
            # The following assumes the data is not batched yet and we are dealing with a 1-D tensor, a single example
            # last_item_index = tf.expand_dims(tf.size(features['asin'])//2 - 1, axis=0)  # Needs to be 1-D tensor
            last_item_index = tf.expand_dims(tf.size(features['asin']) - 1, axis=0)  # Needs to be 1-D tensor
            features['asin'], labels = mask_items(features['asin'], last_item_index)
        else:
            features['asin'], labels = random_item_mask(
                item_list=features['asin'],
                masked_percentage=MASKED_PERCENTAGE,
                max_masked=1
            )
    else:
        raise ValueError(f'Unrecognized mode: {mode}')

    # features['orig_labels'] = labels
    features['labels'] = label_map(labels)

    return features


def create_cloze_dataset(source, mode, batch_size, target_vocab_file):
    """

    Args:
        source: If str, must be the path to data files (not directory). For example data/*.tfrecord
        mode:
        batch_size:

    Returns:

    """
    if isinstance(source, str):
        filenames = tf.data.Dataset.list_files(source)
        dataset = tf.data.TFRecordDataset(filenames=filenames)

        feature_spec = {
            'reviewerID': tf.io.FixedLenFeature([], dtype=tf.string),
            'asin': tf.io.VarLenFeature(dtype=tf.string),
            'unixReviewTime': tf.io.VarLenFeature(dtype=tf.int64)
        }

        def parse_fn(ex):
            return parse_examples(ex, feature_spec)
        dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    elif callable(source):
        # ToDo: It might be possible to define this only once (like above) and deduce type and shape from that
        #  so that we can unify this with the one above
        data_types = {
            'asin': tf.string,
            'reviewerID': tf.string,
            'unixReviewTime': tf.int64,
            # 'labels': tf.int64
        }
        tensor_shapes = {
            'asin': tf.TensorShape([None]),
            'reviewerID': tf.TensorShape([]),
            'unixReviewTime': tf.TensorShape([None]),
            # 'labels': tf.TensorShape([])
        }

        dataset = tf.data.Dataset.from_generator(source, output_types=data_types, output_shapes=tensor_shapes)

    else:
        raise TypeError('Source must be either str or callable.')

    # Shuffle then repeat. Batch should always be after these two (https://stackoverflow.com/a/49916221/4936825)
    if mode == modes.TRAIN:
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    dataset = dataset.repeat(None)

    # This lookup table is needed to map label tokens to integers. It should be created outside the function
    # that will be passed to dataset.map. Creating it inside that function will result in a TF warning
    label_vocab = load_vocabulary(target_vocab_file)
    values = tf.convert_to_tensor(range(len(label_vocab)), dtype=tf.int64)
    initializer = tf.lookup.KeyValueTensorInitializer(keys=label_vocab, values=values)
    label_lookup = tf.lookup.StaticVocabularyTable(initializer, num_oov_buckets=1)

    item_mask = functools.partial(cloze_data_prep, mode=mode, label_lookup_table=label_lookup)

    dataset = dataset.map(item_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes={  # Pad all to longest in the batch
            'reviewerID': [],
            'asin': [None],
            'unixReviewTime': [None],
            'labels': [None],
            # 'orig_labels': [None]
        },
        padding_values={
            'reviewerID': INPUT_PADDING_TOKEN,
            'asin': INPUT_PADDING_TOKEN,
            'unixReviewTime': tf.cast(INPUT_PAD, tf.int64),
            'labels': LABEL_PAD,
            # 'orig_labels': INPUT_PADDING_TOKEN
        }
    )

    def pop_labels(features):
        labels = features.pop('labels')
        return features, labels

    dataset = dataset.map(pop_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # TODO: TensorFlow docs mention wrapping map functions in py_function. I don't know if it is for optimization or
    #  just for eager mode compatibility. But check that out.

    # TODO: Figure out caching. This doesn't work right now.
    # dataset = dataset.cache()  # Cache to memory to speed up subsequent reads

    # TODO: Consider interleave as well
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


if __name__ == '__main__':
    N_ITEMS = 10
    input_dir = '../data/amazon_beauty_bert4rec'
    # data_gen = ClickStreamGenerator(
    #     n_items=N_ITEMS,
    #     n_events=10,
    #     session_cohesiveness=5,
    #     write_vocab_files=True,
    #     vocab_dir='../data/simulated/vocabs'
    # )
    item_vocab_file = os.path.join(input_dir, 'vocabs/item_vocab.txt')
    data = create_cloze_dataset(
        source=input_dir + '/*.tfrecord',
        mode='TRAIN',
        batch_size=2,
        target_vocab_file=item_vocab_file
    )

    sequential_input_config = {
        'items': ['asin'],
        # 'events': ['seq_1_events', 'seq_2_events']
    }

    feature_vocabularies = {
        'items': item_vocab_file,
        # 'events': 'data/vocabs/event_vocab.txt'
    }

    embedding_dims = {
        'items': 4,
        # 'events': 2
    }

    final_layers_dims = [10, 5]
    softmax_head = SoftMaxHead(dense_layer_dims=final_layers_dims, output_vocab_size=N_ITEMS)

    # clickstream_model = ClickstreamTransformer(
    #     sequential_input_config=sequential_input_config,
    #     feature_vocabs=feature_vocabularies,
    #     embedding_dims=embedding_dims,
    #     head_unit=softmax_head,
    #     value_to_head=INPUT_MASKING_TOKEN,
    #     num_encoder_layers=1,
    #     num_attention_heads=1,
    #     dropout_rate=0.1
    # )
