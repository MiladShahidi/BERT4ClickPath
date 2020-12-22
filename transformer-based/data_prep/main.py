import tensorflow as tf
import pandas as pd
import numpy as np
import data_utils

def process_raw_data(filename, columns_name, pickle_file, limit_user_item=10):
    """
    Read transactional data and create baskets that incluses a list of items for each session.
    Example:
    Given the following Pandas DataFrame:

           userID  date           itemID
        0   1           11/1/2000            131
        1   1           11/1/2000            148
        2   1           11/1/2000            123
        3   1           13/2/2000            198
        4   1           13/2/2000            143

    Converted to:

            userID      basket
        0   1           [131, 148, 123]
        1   1           [198, 143]

    Args:
        filename: path (including name of the csv file) to the transaction data in csv format
        columns_name: List of original column names in the transaction dataset which correspond to
                        userID, itemID, time_of_transaction
        pickle_file: path (including pickle name) to save dictionary of item vocabulary
        limit_user_item: int : threshold to eliminate items that have been purchased by less that this value and also
                                         to eliminate the user that has purchased less than limit_user_item itmes

    Returns:
        A Pandas DataFrame
    """
    df = pd.read_csv(filename)
    df = df[columns_name]
    # df = df.head(10000)
    df.columns = ['userID', 'itemID', 'timestamp']


    # remove the user that has purchased less than 10 items
    df1 = df.groupby('userID').agg({'itemID': 'count'}).rename(
        columns={'itemID': 'num_purchased_item'}).reset_index()
    df1 = df1[df1['num_purchased_item'] > limit_user_item]
    df = pd.merge(df1['userID'], df, on='userID', how='left')

    # remove the products that has been purchased by less than 10 customers
    df1 = df.groupby('itemID').agg({'userID': 'count'}).rename(
        columns={'userID': 'num_purchased'}).reset_index()
    df1 = df1[df1['num_purchased'] > limit_user_item]
    df = pd.merge(df1['itemID'], df, on='itemID', how='left')

    # convert PRODUCT_ID to categorical variable
    df['itemID'] = df['itemID'].astype('category')
    dic_product_id = {k + 1: v for k, v in enumerate(df['itemID'].cat.categories)}
    df['itemID'] = df['itemID'].cat.codes + 1  # start the item index from 1
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('datetime')

    # with open(pickle_file, 'wb') as handle:
    #     pickle.dump(dic_product_id, handle)

    baskets = df.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    baskets = baskets[['userID', 'itemID']]

    return baskets


def pandas_to_seq_example(df, group_id_column):
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
    def to_seq_example(collected_row):
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

        sequence_features_dict = {k: tf.train.FeatureList(feature=[data_utils.to_feature(element) for element in v])
                                  for k, v in sequence_features_dict.items()}
        sequence_features_dict = tf.train.FeatureLists(feature_list=sequence_features_dict)

        context_features_dict = {k: data_utils.to_feature(v) for k, v in context_features_dict.items()}
        seq_example = tf.train.SequenceExample(
            context=tf.train.Features(feature=context_features_dict),
            feature_lists=sequence_features_dict
        )

        return seq_example

    collected_df = pd.DataFrame()
    for col in df.columns:
        if col != group_id_column:
            collected_df[col] = df.groupby(group_id_column)[col].apply(list)

    return collected_df.apply(to_seq_example, axis=1).to_list()


def write_to_tfrecord(data, shard_name_temp, records_per_shard=10**4):
    shard_boundaries = [k * records_per_shard for k in range(len(data) // records_per_shard + 1)]
    if shard_boundaries[-1] < len(data):
        shard_boundaries.append(len(data))  # in case the number of records is not an exact multiple of shard size
    num_shards = len(shard_boundaries) - 1
    for i, (shard_start, shard_end) in enumerate(zip(shard_boundaries, shard_boundaries[1:])):
        with tf.io.TFRecordWriter(shard_name_temp % (i, num_shards)) as writer:
            for record in data[shard_start:shard_end]:
                writer.write(record.SerializeToString())


if __name__ == '__main__':

    transaction_dataset = True
    if transaction_dataset:
        df = process_raw_data('../data/raw_data/ta-fen-transaction.csv',
                              ['CUSTOMER_ID', 'PRODUCT_ID', 'TRANSACTION_DT'],
                              '../data/item_vocab.pickle', limit_user_item=10)
    else:
        df = pd.DataFrame({
            'userID': [1, 1, 2, 2, 3],
            'basket': [list(np.random.randint(100, 200, size=np.random.randint(low=1, high=10))) for _ in range(5)]
        })

    data = pandas_to_seq_example(df, 'userID')

    write_to_tfrecord(data, 'test_data_%d_of_%d.tfrecord')

    # Reading it back

    tf_dataset = tf.data.TFRecordDataset('test_data_0_of_1.tfrecord')

    context_feature_spec = {
        'userID': tf.io.FixedLenFeature([], tf.int64),
    }
    sequence_feature_spec = {
        'basket': tf.io.VarLenFeature(tf.int64)
    }

    def parse_fn(x):
        parsed_context, parsed_sequence = tf.io.parse_single_sequence_example(
            serialized=x,
            context_features=context_feature_spec,
            sequence_features=sequence_feature_spec
        )
        # A VarLenFeature is always parsed to a SparseTensor
        features_dict = {**parsed_sequence, **parsed_context}
        for key in features_dict.keys():
            if isinstance(features_dict[key], tf.SparseTensor):
                features_dict[key] = tf.sparse.to_dense(features_dict[key])

        return features_dict

    tf_dataset = tf_dataset.map(parse_fn)

    INPUT_PAD = 0
    INPUT_PADDING_TOKEN = '[PAD]'

    tf_dataset = tf_dataset.padded_batch(
        batch_size=3,
        padded_shapes={  # Pad all to longest in batch
            'userID': [],
            'basket': [None, None]
        },
        padding_values={
            'userID': tf.cast(INPUT_PAD, tf.int64),
            'basket': tf.cast(INPUT_PAD, tf.int64)
        }
    )
    # from pprint import pprint
    for x in tf_dataset:
        pprint(x)
