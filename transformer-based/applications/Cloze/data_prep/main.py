import tensorflow as tf
import pandas as pd
import numpy as np
from sequence_transformer.data_prep import data_utils
import json
import gzip
import os


def parse(path, use_columns=None):
    g = gzip.open(path, 'rb')
    for l in g:
        record = json.loads(l)
        if use_columns is not None:
            record = {k: v for k, v in record.items() if k in use_columns}
        yield record


def get_pandas_df(path, use_columns=None, n_rows=None):
    i = 0
    df = {}
    for d in parse(path, use_columns):
        df[i] = d
        i += 1
        if n_rows is not None:
            if i == n_rows:
                break

    return pd.DataFrame.from_dict(df, orient='index')


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    MIN_ITEM = 4

    output_path = '../data'

    print('Reading data...')
    df = get_pandas_df(path='../raw_data/reviews_Beauty.json.gz',
                       use_columns=['reviewerID', 'asin', 'unixReviewTime'],
                       n_rows=1000)

    # df = pd.DataFrame({
    #     'reviewerID': [1, 2, 1, 2, 3],
    #     'asin': ['a_late', 'b_late', 'a_early', 'c_early', 'c'],
    #     'unixReviewTime': [6, 2, 4, 1, 3]
    # })

    min_item_filter = df.groupby('reviewerID')['asin'].transform('count').ge(MIN_ITEM)
    df = df[min_item_filter]

    item_vocab = pd.unique(df['asin'])
    vocab_path = os.path.join(output_path, 'vocabs')
    os.makedirs(vocab_path, exist_ok=True)
    with open(os.path.join(vocab_path, 'item_vocab.txt'), 'w') as f:
        f.writelines('\n'.join(item_vocab))

    print('Converting to TF Examples...')
    df = df.sort_values(['unixReviewTime'])
    tf_data = data_utils.pandas_to_tf_example_list(df, group_id_column='reviewerID')

    print('Writing TFRecord files...')
    data_utils.write_to_tfrecord(tf_data, path=output_path, filename_prefix='amazon_beauty')

    # reading it back
    if False:
        files = [os.path.join(output_path, filename) for filename in tf.io.gfile.listdir(output_path)]
        feature_spec = {
            'reviewerID': tf.io.FixedLenFeature([], tf.string),
            'asin': tf.io.VarLenFeature(tf.string),
            'unixReviewTime': tf.io.VarLenFeature(tf.int64)
        }

        dataset = tf.data.TFRecordDataset(files)

        for x in dataset.take(1):
            ex = tf.io.parse_single_example(x, feature_spec)
            print(ex)
