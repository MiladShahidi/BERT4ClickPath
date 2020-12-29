import tensorflow as tf
from sequence_transformer.constants import INPUT_PADDING_TOKEN, LABEL_PAD, INPUT_PAD
from data_generator import ReturnsDataGen
from sequence_transformer.clickstream_model import ClickstreamModel
import os


def parse_examples(serialized_example, feature_spec):
    features = tf.io.parse_example(serialized=serialized_example, features=feature_spec)

    # A VarLenFeature is always parsed to a SparseTensor
    for key in features.keys():
        if isinstance(features[key], tf.SparseTensor):
            features[key] = tf.sparse.to_dense(features[key])

    return features


def random_choice(x, size, axis=0):
    """
    This is the equivalent of np.random.choice. But it always returns unique choices, i.e. keep_features=False in numpy
    """
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices)[:size]
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

    basket = tf.where(labels == 1, x=positive_samples, y=negative_samples)  # Populate the basket

    return {
        'page_view_product_skn_ids': tf.squeeze(masked_item_list, -1),
        'basket_product_id': basket,
        'labels': labels,
    }


def create_tf_dataset(source, training, batch_size):

    if isinstance(source, str):
        filenames = tf.data.Dataset.list_files(source)
        dataset = tf.data.TFRecordDataset(filenames=filenames)

        feature_spec = {
            'reviewerID': tf.io.VarLenFeature(dtype=tf.string),
            'asin': tf.io.VarLenFeature(dtype=tf.string),
            'unixReviewTime': tf.io.VarLenFeature(dtype=tf.float32)
        }

        def parse_fn(ex):
            return parse_examples(ex, feature_spec)
        dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    elif callable(source):
        pass
        # ToDo: It might be possible to define this only once (like above) and deduce type and shape from that
        #  so that we can unify this with the one above
        # data_types = {
        #     'seq_1_items': tf.string,
        #     'seq_1_events': tf.string,
        #     'seq_2_items': tf.string,
        #     'seq_2_events': tf.string,
        #     'side_feature_1': tf.float32,
        #     'label': tf.float32  # Label needs to be float for calculations in the loss function. int won't work
        # }
        # tensor_shapes = {
        #     'seq_1_items': tf.TensorShape([None]),
        #     'seq_1_events': tf.TensorShape([None]),
        #     'seq_2_items': tf.TensorShape([None]),
        #     'seq_2_events': tf.TensorShape([None]),
        #     'side_feature_1': tf.TensorShape([]),
        #     'label': tf.TensorShape([None])
        # }
        #
        # dataset = tf.data.Dataset.from_generator(source,
        #                                          output_types=data_types,
        #                                          output_shapes=tensor_shapes)

    else:
        raise TypeError('Source must be either str or callable.')

    # Shuffle then repeat. Batch should always be after these two (https://stackoverflow.com/a/49916221/4936825)
    if training:
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    dataset = dataset.repeat(None)

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes={  # Pad all to longest in batch
            'reviewerID': [],
            'asin': [None],
            'unixReviewTime': [None]
        },
        padding_values={
            'reviewerID': INPUT_PADDING_TOKEN,
            'asin': INPUT_PADDING_TOKEN,
            'unixReviewTime': INPUT_PAD
        }
    )

    # TODO: TensorFlow docs mention wrapping map functions in py_function. I don't know if it is for optimization or
    #  just for eager mode compatibility. But check that out.

    # TODO: Figure out caching. This doesn't work right now.
    # dataset = dataset.cache()  # Cache to memory to speed up subsequent reads

    def pop_labels(feature_dict):
        labels = feature_dict.pop('label')
        return feature_dict, labels

    dataset = dataset.map(pop_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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
    # data_gen = ReturnsDataGen(
    #     n_items=1000,
    #     n_events=10,
    #     session_cohesiveness=5,
    #     positive_rate=0.5,
    #     write_vocab_files=True,
    #     vocab_dir='../data/vocabs'
    # )

    data = create_tf_dataset(
        source='../data',
        training=True,
        batch_size=2
    )

    sequential_input_config = {
        'items': ['seq_1_items', 'seq_2_items'],
        # 'events': ['seq_1_events', 'seq_2_events']
    }

    feature_vocabularies = {
        'items': '../data/vocabs/item_vocab.txt',
        # 'events': '../data/vocabs/event_vocab.txt'
    }

    embedding_dims = {
        'items': 4,
        # 'events': 2
    }

    returns_model = ClickstreamModel(
        sequential_input_config=sequential_input_config,
        feature_vocabs=feature_vocabularies,
        embedding_dims=embedding_dims,
        segment_to_output=2,
        num_encoder_layers=1,
        num_attention_heads=1,
        dropout_rate=0.1,
        final_layers_dims=[10, 5]
    )

    for x, y in data.take(1):
        print(x)
        # print_features(x)
        print('Label:')
        print(y)
