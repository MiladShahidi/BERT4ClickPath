import tensorflow as tf
import numpy as np
import pandas as pd
import os


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


def pandas_to_tf_example_list(df, group_id_column):
    # ToDo: It would be nice to add a sort_by argument to allow within group sorting, in case data includes a timestamp
    def to_tf_example(collected_row):
        row_dict = collected_row.to_dict()
        # This is what the DataFrame has been grouped by
        key_feature = {group_id_column: collected_row.name}  # df.name is the value of group_id for this group
        features_dict = {**row_dict, **key_feature}

        return encode_tf_example(features_dict)

    collected_df = pd.DataFrame()
    for col in df.columns:
        if col != group_id_column:
            collected_df[col] = df.groupby(group_id_column)[col].apply(list)

    return collected_df.apply(to_tf_example, axis=1).to_list()


def pandas_to_tf_seq_example_list(df, group_id_column):
    """
    This function converts a pandas DataFrame into a list of Tensorflow SequenceExample objects.
    It performs a groupby using `group_id_column` and collect the values of all other columns into lists (simillar to
    PySpark's `collect_list`). Then the `group_id_column` column will be put into the `context` component of the
    SequenceExample object and all other columns (each of which is now a list) will be put in the `feature_list`
    component.

    Example:
    Given the following Pandas DataFrame:

           id  int_feature           basket
        0   1           10            [131]
        1   1           11       [152, 148]
        2   2           12  [161, 106, 134]
        3   2           13       [171, 123]
        4   3           14            [123]

    The output corresponding to each group will be a SequenceExample object which consists of two components:
        1) context
        2) feature_lists

    For example, the output for id=2 will look like:

        context {
          feature {
            key: "id"
            value {
              int64_list {
                value: 2
              }
            }
          }
        }
        feature_lists {
          feature_list {
            key: "basket"
            value {
              feature {
                int64_list {
                  value: 161
                  value: 106
                  value: 134
                }
              }
              feature {
                int64_list {
                  value: 171
                  value: 123
                }
              }
            }
          }
          feature_list {
            key: "int_feature"
            value {
              feature {
                int64_list {
                  value: 12
                }
              }
              feature {
                int64_list {
                  value: 13
                }
              }
            }
          }
        }

    Notice that the column that contained a list per row, is converted into a list of lists (basket here).

    Args:
        df: A Pandas DataFrame.
        group_id_column: The name of the column to be used for groupby.

    Returns:
        A list of tf.train.SequenceExample objects each element of which corresponds to a group in
            df.groupby(group_id_column)
    """
    def to_tf_seq_example(collected_row):
        row_dict = collected_row.to_dict()
        # This is what the DataFrame has been grouped by
        context_features_dict = {group_id_column: collected_row.name}
        sequence_features_dict = {}
        for feature_name, feature_value in row_dict.items():
            # Since we receive one row of the results of a groupby operation, each column is a list. That much
            # is certain. However, some columns are a 1-D list while other are nested 2-D lists.
            if isinstance(feature_value[0], list):  # Examine one value to see whether this is a list or a list of lists
                sequence_features_dict[feature_name] = feature_value
            else:
                context_features_dict[feature_name] = feature_value

        sequence_features_dict = {k: tf.train.FeatureList(feature=[to_feature(element) for element in v])
                                  for k, v in sequence_features_dict.items()}
        sequence_features_dict = tf.train.FeatureLists(feature_list=sequence_features_dict)

        context_features_dict = {k: to_feature(v) for k, v in context_features_dict.items()}
        seq_example = tf.train.SequenceExample(
            context=tf.train.Features(feature=context_features_dict),
            feature_lists=sequence_features_dict
        )

        return seq_example

    collected_df = pd.DataFrame()
    for col in df.columns:
        if col != group_id_column:
            collected_df[col] = df.groupby(group_id_column)[col].apply(list)

    # ToDo: This function decides howto split features between the "context" and the "sequence" components of the
    #  sequence example. It is probably better to either add this as an argument so that the caller can specify that
    #  which would make the caller code more readable, or have this function return the resulting allocation like
    #  component_features = ['context_feature_1' , 'context_feature_2']
    #  sequence_features = ['seq_feature_1', 'seq_feature_2']
    return collected_df.apply(to_tf_seq_example, axis=1).to_list()


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


def write_to_tfrecord(data, path, filename_prefix, records_per_shard=10 ** 4):
    """
    Writes Tensorflow examples to tfrecord files.

    Args:
        data: A list of tf.Example or tf.sequence_example objects.
        filename_prefix: Prefix for filename(s). In case data is sharded, file numbers will be appended to this prefix.
        records_per_shard: Number of record to write in each file.

    Returns:
        None
    """
    shard_boundaries = [k * records_per_shard for k in range(len(data) // records_per_shard + 1)]
    if shard_boundaries[-1] < len(data):
        shard_boundaries.append(len(data))  # in case the number of records is not an exact multiple of shard size
    num_shards = len(shard_boundaries) - 1
    for i, (shard_start, shard_end) in enumerate(zip(shard_boundaries, shard_boundaries[1:])):
        os.makedirs(path, exist_ok=True)
        filename = filename_prefix + f'_{i}_of_{num_shards}.tfrecord'
        filename = os.path.join(path, filename)
        with tf.io.TFRecordWriter(filename) as writer:
            for record in data[shard_start:shard_end]:
                writer.write(record.SerializeToString())
