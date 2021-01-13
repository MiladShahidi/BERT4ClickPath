import tensorflow as tf
from sequence_transformer.constants import INPUT_PADDING_TOKEN, LABEL_PAD, INPUT_PAD, INPUT_MASKING_TOKEN
from data_generator import ClickStreamGenerator
from sequence_transformer.clickstream_model import ClickstreamModel
import os
from applications.Cloze.cloze_constants import MAX_MASKED_ITEMS, MASKED_PERCENTAGE
from sequence_transformer.head import SoftMaxHead
from sequence_transformer.clickstream_model import TokenMapper


def parse_examples(serialized_example, feature_spec):
    features = tf.io.parse_example(serialized=serialized_example, features=feature_spec)

    # A VarLenFeature is always parsed to a SparseTensor
    for key in features.keys():
        if isinstance(features[key], tf.SparseTensor):
            features[key] = tf.sparse.to_dense(features[key])

    return features


def random_choice(x, size, axis=0, preserve_order=True):
    """
    This is the equivalent of np.random.choice. But it always returns unique choices, i.e. keep_features=False in numpy
    """
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices)[:size]
    if preserve_order:
        sample_index = tf.sort(sample_index)

    sample = tf.gather(x, sample_index, axis=axis)

    return sample, sample_index


def format_labels(feature_dict):
    """
        This mapping is applied after batching. It adds padding (-1) to the left of labels (which are associated with
        seq_2 items). The seq_1 component does not have labels and should be assigned pads instead of labels.
    """
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


def random_item_mask(item_list, masked_percentage=MASKED_PERCENTAGE, max_masked=MAX_MASKED_ITEMS):
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
    masked_items, mask_index = random_choice(item_list, n_masked)

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


def create_tf_dataset(source, is_training, batch_size):
    """

    Args:
        source: If str, must be the path to data files (not directory). For example data/*.tfrecord
        is_training:
        batch_size:

    Returns:

    """
    if isinstance(source, str):
        # files = [os.path.join(source, filename) for filename in tf.io.gfile.listdir(source)]
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
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    dataset = dataset.repeat(None)

    # ToDo: This is bad. The vocab address should not be a literal string in here
    label_mapper = TokenMapper(vocabularies={'labels': 'data/simulated/vocabs/item_vocab.txt'})

    def item_mask(features):
        features['asin'], labels = random_item_mask(
            item_list=features['asin'],
            masked_percentage=MASKED_PERCENTAGE,
            max_masked=MAX_MASKED_ITEMS
        )
        features['orig_labels'] = labels
        features['labels'] = label_mapper({'labels': labels})['labels']
        return features

    dataset = dataset.map(item_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes={  # Pad all to longest in batch
            'reviewerID': [],
            'asin': [None],
            'unixReviewTime': [None],
            'labels': [None],
            'orig_labels': [None]
        },
        padding_values={
            'reviewerID': INPUT_PADDING_TOKEN,
            'asin': INPUT_PADDING_TOKEN,
            'unixReviewTime': tf.cast(INPUT_PAD, tf.int64),
            'labels': tf.cast(LABEL_PAD, tf.int64),
            'orig_labels': INPUT_PADDING_TOKEN
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


def print_features(x, select=None):
    for k in x:
        if (select is None) or (k in select):
            print(k)
            print('\t', x[k])
            print('*'*80)


def dataset_benchmark(dataset, n_steps):
    import time
    start_time = time.perf_counter()
    for x in dataset.take(n_steps):
        time.sleep(0.01)  # train step
    return time.perf_counter() - start_time


if __name__ == '__main__':
    N_ITEMS = 40
    data_gen = ClickStreamGenerator(
        n_items=N_ITEMS,
        n_events=10,
        session_cohesiveness=5,
        write_vocab_files=True,
        vocab_dir='data/vocabs'
    )

    data = create_tf_dataset(
        source=data_gen,
        is_training=True,
        batch_size=10
    )

    sequential_input_config = {
        'items': ['asin'],
        # 'events': ['seq_1_events', 'seq_2_events']
    }

    feature_vocabularies = {
        'items': 'data/vocabs/item_vocab.txt',
        # 'events': 'data/vocabs/event_vocab.txt'
    }

    embedding_dims = {
        'items': 4,
        # 'events': 2
    }

    final_layers_dims = [10, 5]
    softmax_head = SoftMaxHead(dense_layer_dims=final_layers_dims, output_vocab_size=3)

    clickstream_model = ClickstreamModel(
        sequential_input_config=sequential_input_config,
        feature_vocabs=feature_vocabularies,
        embedding_dims=embedding_dims,
        head_unit=softmax_head,
        value_to_head=INPUT_MASKING_TOKEN,
        num_encoder_layers=1,
        num_attention_heads=1,
        dropout_rate=0.1
    )

    for x, y in data.take(1):
        print_features(x)
        print('*'*80)
        y_hat = clickstream_model(x)
        print(y_hat)
        print('*'*80)
        print('Label:')
        print(y)

        from sequence_transformer.training_utils import MaskedLoss
        loss_fn = MaskedLoss(tf.keras.backend.sparse_categorical_crossentropy, label_pad=tf.cast(LABEL_PAD, tf.int64))
        loss = loss_fn(y_true=y, y_pred=y_hat)
        print(loss)
