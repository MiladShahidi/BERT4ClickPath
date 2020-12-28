import tensorflow as tf
import numpy as np
import pandas as pd


def to_feature(value):
    """Returns a feature from a value"""
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    if isinstance(value, list):
        sample_value = value[0]
    else:
        sample_value = value
        value = [value]

    # TODO: This type checking feels hacky. Are np numerical types the same on all machines and Operating Systems?
    # This used to be np.int64, but it didn't work with Pandas dataframes.
    if isinstance(sample_value, np.int) or isinstance(sample_value, np.int32) or isinstance(sample_value, np.int64):
        return _int_feature(value)
    # This used to be np.float32, but it didn't work with Pandas dataframes.
    elif isinstance(sample_value, np.float) or isinstance(sample_value, np.float32):
        return _float_feature(value)
    elif isinstance(sample_value, bytes):
        return _bytes_feature(value)
    elif isinstance(sample_value, str):
        value = [v.encode('utf-8') for v in value]
        return _bytes_feature(value)
    else:
        raise Exception("Encountered unsupported type {}".format(type(sample_value)))


def encode_tf_example(raw_features):
    features = {
        feature_key: to_feature(feature_value) for feature_key, feature_value in raw_features.items()
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def to_example_in_example(grouped_data, encode_example=False):
    """
        grouped_data: a tuple
            The first element of tuple is what it is grouped by, which here are
            the context features. The second element is a list of dicts, each
            of which represent one row of the data. Example:
            (
                {'userId': b'5448543647176335931'},
                [
                    {'itemId': b'299826767', 'score': 0.8023795, 'userId': b'5448543647176335931'},
                    {'itemId': b'299837992', 'score': 0.7049436, 'userId': b'5448543647176335931'},
                    {'itemId': b'299925700', 'score': 0.6766877, 'userId': b'5448543647176335931'}
                ]
            )
        This is how apache-beam represents (grouped) data

        For an example of the Example-in-Example data format see `tfr.data.parse_from_example_in_example`
        For details on tf.Example see:
        https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564

        encode_example: Beam has its own ProtoExampleCoder which will encode a tf.Example. So this is set to False by
        default. If true it will return a tf.Example instead of dict of features
    """
    context_features_dict, example_dictlist = grouped_data
    # First, remove the keys (columns of data) that are already present in context_features
    for example_dict in example_dictlist:
        for key in context_features_dict:
            example_dict.pop(key, None)

    serialized_context = encode_tf_example(context_features_dict).SerializeToString()

    serialized_examples = [encode_tf_example(example_dict).SerializeToString() for example_dict in example_dictlist]

    if encode_example:
        eie_features = {
            "serialized_context": tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_context])),
            "serialized_examples": tf.train.Feature(bytes_list=tf.train.BytesList(value=serialized_examples)),
        }
        return tf.train.Example(features=tf.train.Features(feature=eie_features))
    else:
        return {
            "serialized_context": [serialized_context],
            "serialized_examples": serialized_examples
        }


def to_sequence_example(grouped_data, encode_example=False):
    """
        grouped_data: a tuple
            The first element of tuple is what it is grouped by, which here are
            the context features. The second element is a list of dicts, each
            of which represent one row of the data. Example:
            (
                {'userId': b'5448543647176335931'},
                [
                    {'itemId': b'299826767', 'score': 0.8023795, 'userId': b'5448543647176335931'},
                    {'itemId': b'299837992', 'score': 0.7049436, 'userId': b'5448543647176335931'},
                    {'itemId': b'299925700', 'score': 0.6766877, 'userId': b'5448543647176335931'}
                ]
            )
        This is how apache-beam represents (grouped) data

        For an example of the Example-in-Example data format see `tfr.data.parse_from_example_in_example`
        For details on tf.Example see:
        https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564

        encode_example: Beam has its own ProtoExampleCoder which will encode a tf.Example. So this is set to False by
        default. If true it will return a tf.Example instead of dict of features
    """
    context_features_dict, example_dictlist = grouped_data
    # First, remove the keys (columns of data) that are already present in context_features
    for example_dict in example_dictlist:
        for key in context_features_dict:
            example_dict.pop(key, None)

    serialized_context = encode_tf_example(context_features_dict).SerializeToString()

    serialized_examples = [encode_tf_example(example_dict).SerializeToString() for example_dict in example_dictlist]

    if encode_example:
        eie_features = {
            "serialized_context": tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_context])),
            "serialized_examples": tf.train.Feature(bytes_list=tf.train.BytesList(value=serialized_examples)),
        }
        return tf.train.Example(features=tf.train.Features(feature=eie_features))
    else:
        return {
            "serialized_context": [serialized_context],
            "serialized_examples": serialized_examples
        }


def pandas_train_test_split(df, train_size, context_feature_name):
    def mark_training_samples(grouped_df):
        indices = np.random.choice(grouped_df.index, size=train_size, replace=False)
        train_indices = grouped_df.index.isin(indices)
        grouped_df['train'] = train_indices
        return grouped_df

    df = df.groupby(context_feature_name).apply(mark_training_samples)
    train_df = df[df['train']].drop('train', axis=1)
    eval_df = df[~df['train']].drop('train', axis=1)
    return train_df, eval_df


def write_to_tfrecord(data, shard_name_temp, records_per_shard=10**4):
    shard_boundaries = [k * records_per_shard for k in range(len(data) // records_per_shard + 1)]
    if shard_boundaries[-1] < len(data):
        shard_boundaries.append(len(data))  # in case the number of records is not an exact multiple of shard size
    num_shards = len(shard_boundaries) - 1
    for i, (shard_start, shard_end) in enumerate(zip(shard_boundaries, shard_boundaries[1:])):
        with tf.io.TFRecordWriter(shard_name_temp % (i, num_shards)) as writer:
            for record in data[shard_start:shard_end]:
                writer.write(record.SerializeToString())
