import tensorflow as tf
from constants import MODEL_MAX_SESSION_LEN, MODEL_MAX_BASKET_SIZE
# from constants import INPUT_MAX_BASKET_SIZE, INPUT_MAX_SESSION_LEN
# from constants import INPUT_MIN_BASKET_SIZE, INPUT_MIN_SESSION_LEN
from constants import INPUT_PADDING_TOKEN, UNKNOWN_EVENT_OR_ITEM, LABEL_PADDING_VALUE, INPUT_MASKING_TOKEN
from constants import CLASSIFICATION_TOKEN, SEPARATOR_TOKEN, SYNTHETIC_POSITIVE_SAMPLE_RATE, NUM_RESERVED_TOKENS
from constants import INPUT_MIN_SESSION_LEN
import tensorflow_transform as tft
import os


PRETRAINING_MASKING_PERCENTAGE = 0.2


def load_vocabulary(vocab_file):
    with tf.io.gfile.GFile(vocab_file, 'r') as f:
        vocab = tf.strings.strip(f.readlines())

    return vocab


def feature_transformation(features, vocab_file):
    # This lambda is just for conciseness and convenience
    vocab_lookup = lambda word: tft.apply_vocabulary(
        x=word,
        deferred_vocab_filename_tensor=vocab_file,
        default_value=UNKNOWN_EVENT_OR_ITEM)

    features['session_and_basket'] = vocab_lookup(features['session_and_basket'])

    # Alternative
    # from tensorflow.python.ops.lookup_ops import TextFileIdTableInitializer
    # table = tf.lookup.StaticVocabularyTable(
    #     TextFileIdTableInitializer(vocab_file), 1)
    # table._initialize()
    # features['session_and_basket'] = table.lookup(features['session_and_basket'])

    return features


def truncate_sequences(features, max_sess_len):
    # TODO: Once an input schema is defined, we should be able to make this smarter and more general.
    #  At least avoid string literals for names
    for feature_name in ['session_and_basket', 'labels']:
        features[feature_name] = features[feature_name][-max_sess_len:]  # Takes the last MODEL_MAX_SESSION_LEN entries

    # Basket sequences
    # for feature_name in ['basket_product_id', 'labels']:
    #     features[feature_name] = features[feature_name][-max_basket_size:]  # Takes the last MODEL_MAX_BASKET_SIZE entries

    return features


def make_masked_exmples(feature_dict, negative_sample_pool):
    item_list = feature_dict['page_view_product_skn_ids']
    session_len = tf.cast(tf.size(item_list), tf.float32)
    basket_size = tf.cast(tf.multiply(session_len, PRETRAINING_MASKING_PERCENTAGE), dtype=tf.int32)
    basket_size = tf.clip_by_value(basket_size, clip_value_min=0, clip_value_max=MODEL_MAX_BASKET_SIZE)

    # Randomly pick basket_size items to mask from the list of items
    positive_samples, mask_index = random_choice(item_list, basket_size)

    # Mask the chosen items by replacing them with the padding constant
    mask = tf.fill(dims=tf.shape(mask_index), value=INPUT_MASKING_TOKEN)

    # scatter_update requires the inner most (last) dimension of its inputs to be the same. I'm setting that to one.
    masked_item_list = tf.tensor_scatter_nd_update(
        tensor=tf.reshape(item_list, (-1, 1)),
        indices=tf.reshape(mask_index, (-1, 1)),
        updates=tf.reshape(mask, (-1, 1))
    )

    # # # # # # # # # # # # # # # #
    # Lesson learned about random number generation:
    # I was initially generating random labels as:
    # labels = tf.keras.backend.random_binomial(shape=(basket_size,), p=POSITIVE_EXAMPLE_RATE)
    # But this acts strangely in graph mode (which is how tf.data executes the data pipeline)
    # The mean of 1000 draws from this with p=0.5 is somewhere around 0.7 or 0.8 (instead of 0.5)
    # This is when the same data point is drawn 100 times. But if you draw 1000 different examples from the pipeline
    # I'm not sure how to interpret this other than the fact that this is meant for eager execution and should not be
    # used in graph mode (therefore, should not be part of tf.data pipeline)
    # # # #
    # On the other hand, tf.random.uniform (soon to be deprecated) behaves exactly as expected.
    # According to docs there will soon be a new tf.random.Generator class. (only in tf-nightly at the moment)
    # https://www.tensorflow.org/api_docs/python/tf/random/Generator#uniform
    # for now, tf.compat.v1.random.experimental.get_global_generator does the same thing

    basket_shape = (basket_size,)
    # This will be replaced with tf.random.Generator in the future (currently only available in tf-nightly)
    rand_gen = tf.compat.v1.random.experimental.get_global_generator()
    uniform_randoms = rand_gen.uniform(shape=basket_shape)
    labels = tf.where(uniform_randoms < SYNTHETIC_POSITIVE_SAMPLE_RATE, tf.ones(basket_shape), tf.zeros(basket_shape))

    negative_samples, _ = random_choice(negative_sample_pool, basket_size)

    basket = tf.where(labels == 1, x=positive_samples, y=negative_samples)  # Populate the seq_2

    return {
        'page_view_product_skn_ids': tf.squeeze(masked_item_list, -1),
        'basket_product_id': basket,
        'labels': labels,
    }


def prepare_basket(features):
    """
    This mapping repeats items in the seq_2 according to their quantity. For example:

        {'shipped_quantity': [2 1], 'basket_skn_id': [b'601' b'602']}

    results in:

        'basket_skn_id': [b'601' b'601' b'602']

    It also converts returned_quantity into a binary flag (returned or not) and assigns it to all items of that type
    (if quantity > 1). So, for example:

        {'returned_quantity': [1 0], 'shipped_quantity': [2 1], 'basket_skn_id': [b'601' b'602']}

    is transformed to:

        {'basket_skn_id': [b'601' b'601' b'602'], 'returned': [1 1 0]}

    even though only one of '601' was returned.

    The idea is that the model has no way to know which one of two identical items was returned. Therefore the
    training label should be identical for identical items. So, the model will only predict which item will be
    returned, but not how many of it.

    Args:
        features:

    Returns:

    """
    returned_flag = tf.clip_by_value(features['returned_quantity'], clip_value_min=0, clip_value_max=1)
    returned_flag = tf.cast(returned_flag, tf.float32)  # Loss fn requires label to be float (I think)
    basket = {
        'seq_2': tf.repeat(features['basket_product_id'], features['shipped_quantity']),
        'labels': tf.repeat(returned_flag, features['shipped_quantity'])
    }

    return basket


def create_basket(features, vocab=None, pre_training=False):
    """

    Args:
        features:
        vocab: only required when pre_training=True
        pre_training:

    Returns:

    """
    # Replicate each item (and, for seq_2 items, their corresponding labels) by their quantity. Drop quantity features
    if pre_training:
        assert vocab is not None, 'vocab must be provided when pre_training=True'
        masked_example = make_masked_exmples(features, negative_sample_pool=vocab)
        features.update(masked_example)
    else:
        basket = prepare_basket(features)

    # Following the input format for BERT (except for the SEP at the end)
    # The CLS token in the beginning can be used for sequence classification tasks (will the customer return something?)
    input_sequence = tf.concat([
        [CLASSIFICATION_TOKEN],
        features['page_view_product_skn_ids'],
        [SEPARATOR_TOKEN],
        features['basket_product_id']
    ], axis=0)

    # The labels tensor will have the same length as the input sequence. Input tokens that do not have an associated
    # label will have -1 in place of a label. For example:
    # [ -1, -1, -1, -1,  -1,  1,  0,  0]
    # [CLS, S1, S2, S3, SEP, B1, B2, B3]
    # where S[i] is an item from the seq_1 sequence and B[i] is an item in the seq_2.
    # So, the seq_2 has labels associated with each item. CLS could also have a label for seq. classification
    total_seq_len = tf.shape(input_sequence)[0]
    basket_size = tf.shape(features['basket_product_id'])[0]
    label_padding = tf.repeat(LABEL_PADDING_VALUE, total_seq_len-basket_size)
    labels = tf.concat([label_padding, features['labels']], axis=0)

    features = {
        'order_no': features['order_no'],
        'session_uuid': features['session_uuid'],
        'session_and_basket': input_sequence,
        'labels': labels
    }

    return features


def parse_examples(serialized_example):
    # TODO: There is inconsistency in the raw data in how columns are named. What is called `SKN_ID` in `page_views`
    #  table seems to be `PRODUCT_ID` in the `orders` table. `SKN_ID` from the `page_views` does not match
    #  `PRODUCT_CODE` from `orders`. Because of this, here we are using `basket_product_id` (from `orders` table) and
    #  `page_view_product_skn_ids` (from `page_views`). The names seem to be inconsistent, but these are actually the
    #  columns that match. We need to use proper names to avoid confusion.
    feature_spec = {
        # identifiers
        'order_no': tf.io.FixedLenFeature([], dtype=tf.string),
        'session_uuid': tf.io.FixedLenFeature([], dtype=tf.string),
        # seq_2
        # 'basket_skn_id': tf.io.VarLenFeature(dtype=tf.string),
        'basket_product_id': tf.io.VarLenFeature(dtype=tf.string),  # We don't have a vocabulary for this yet.
        'shipped_quantity': tf.io.VarLenFeature(dtype=tf.int64),
        'returned_quantity': tf.io.VarLenFeature(dtype=tf.int64),
        # sessions
        'page_view_product_skn_ids': tf.io.VarLenFeature(dtype=tf.string),
        'session_page_view_seq': tf.io.VarLenFeature(dtype=tf.int64)
    }

    features = tf.io.parse_example(serialized=serialized_example, features=feature_spec)
    # A VarLenFeature is always parsed to a SparseTensor. So we have to make it dense after parsing
    for key, tensor in features.items():
        if isinstance(tensor, tf.SparseTensor):
            features[key] = tf.sparse.to_dense(tensor)

    return features


def random_choice(x, size, axis=0):
    """
    This is the equivalent of np.random.choice. But it always returns unique choices, i.e. replace=False in numpy
    """
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices)[:size]
    sample = tf.gather(x, sample_index, axis=axis)

    return sample, sample_index


def shuffle_basket(feature_dict):
    raise NotImplementedError('The shuffle_basket function is out of date. Please update before using it.')
    basket = feature_dict['seq_2']
    labels = feature_dict['labels']
    indices = tf.range(start=0, limit=tf.shape(basket)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_basket = tf.gather(basket, shuffled_indices)
    shuffled_labels = tf.gather(labels, shuffled_indices)

    feature_dict['seq_2'] = shuffled_basket
    feature_dict['labels'] = shuffled_labels

    return feature_dict


def load_data(filenames, batch_size, training_mode, vocab_file, vocab, pre_training=False):
    """
    Reading from vocab_file here slows down the pipeline significantly. So, it's better to keep the vocabulary in memory
    and pass it here as a list (vocab). However, we also need the vocab_file because tft.apply_vocabulary needs it
    But consider moving tft.apply_vocabulry (and any other feature transformation step) out of data pipeline, so that
    it will be applied to *all* incoming data (training and prediction) in the same place
    Args:
        filenames: Allows wildcards.
        batch_size:
        training_mode:
        vocab_file:
        vocab: vocabulary as a tensor
        pre_training:
    Returns:

    """
    def _pop_labels(feature_dict):
        labels = feature_dict.pop('labels')
        return feature_dict, labels

    filenames = tf.data.Dataset.list_files(filenames)
    # Shuffle then repeat. Batch should always be after these two (https://stackoverflow.com/a/49916221/4936825)
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    if training_mode:
        dataset = dataset.repeat(None)

    dataset = dataset.map(parse_examples, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def filter_by_session_length(features):
        return tf.size(features['page_view_product_skn_ids']) >= INPUT_MIN_SESSION_LEN

    dataset = dataset.filter(filter_by_session_length)

    # TODO: Tensorflow docs mention wrapping mapping functions in py_function. I don't know if it is for optimization or
    #  only for eager mode compatibility. But check that out.

    dataset = dataset.map(lambda f: create_basket(f, vocab=vocab, pre_training=pre_training),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(lambda f: truncate_sequences(f, MODEL_MAX_SESSION_LEN),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # TODO: Activate this. It will slow down the input pipeline. But it is worth it.
    # dataset = dataset.map(shuffle_basket, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # TODO: Not for the near future. This can save computation and make training faster.
    #  TF 2.2 pads all examples to the longest in the batch by default and doesn't need
    #  the padded_shapes argument. Also, (I might be wrong) the input of the model does not have be the same size across
    #  batches, but only within each batch. So, we don't really need to pad all batches to match the longest example
    #  in the data. The only challenge is that this model has two input components with possibly different lengths.
    #  One thing to try is to concatenate the inputs before padding and insert a [SEP] token between them (as BERT does)
    #  to separate the two parts. Then pad each batch to the longest total length in that batch.
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes={
            'order_no': (),
            'session_uuid': (),
            'session_and_basket': [None],  # Pad to longest in the batch
            # The label will be separated later in the pipeline
            'labels': [None]  # Pad to longest in the batch
        },
        padding_values={
            'order_no': INPUT_PADDING_TOKEN,
            'session_uuid': INPUT_PADDING_TOKEN,
            'session_and_basket': INPUT_PADDING_TOKEN,
            # The label will be separated later in the pipeline
            'labels': LABEL_PADDING_VALUE  # TODO: explain here why 0 cannot be used to pad labels (loss_fn?)
        }, drop_remainder=True)

    # TODO: Important note about this step:
    #  1) Doing feature transformation here is not ideal, because you will have to do the same thing separately in the
    #  prediction pipeline. If it happens inside the model there will be a single point of transformation not two.
    #  2) This will fail in graph mode, i.e. when eager mode is disabled. I don't have a good alternative for it, but
    #  there is one in the comments of the feature_transformation function.

    dataset = dataset.map(lambda features: feature_transformation(features, vocab_file),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def feature_filter(features, keep):
        features = {key: features[key] for key in features.keys() if key in keep}
        return features

    # TODO: Get rid of this step. The schema of what this function returns should be easy to find. This step messes up
    #  the schema. Also, the ids should be passed through the model as instance keys, for debugging
    features_to_keep = ['session_and_basket', 'labels']
    dataset = dataset.map(lambda f: feature_filter(f, features_to_keep),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(_pop_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # TODO: Consider interleave as well
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def print_features(x):
    for k in x:
        print(k)
        print('\t', x[k])


def benchmark(dataset, n_steps):
    import time
    start_time = time.perf_counter()
    for x in dataset.take(n_steps):
        time.sleep(0.01)  # train step
    return time.perf_counter() - start_time


if __name__ == '__main__':

    # tf.compat.v1.disable_eager_execution()
    data_dir = '../data/mock_v1'
    training_data_dir = os.path.join(data_dir, 'training/training_data/part*')
    validation_data_dir = os.path.join(data_dir, 'validation/part*')
    vocab_file = os.path.join(data_dir, 'training/item_vocab/part-00000-17ca410d-8707-4bec-8a3c-e1e914140603-c000.txt')
    # test the dataset here
    vocab = load_vocabulary(vocab_file)

    train_data = load_data(
        filenames=training_data_dir,
        batch_size=2,
        training_mode=False,
        vocab_file=vocab_file,
        vocab=vocab,
        pre_training=True
    )

    # res = benchmark(val_data, 100)
    # print(round(res/100, 3))

    for x, y in train_data.take(1):
        print_features(x)
        print('label: \n\t', y)
