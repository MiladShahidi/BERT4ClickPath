import pandas as pd
from clickstream_transformer import data_utils
import json
import gzip
import os
import numpy as np


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


def read_raw_amazon_data(path, min_item_per_user=None):
    """ Reads data from /json.gz files as formatted in Amazon data repository at https://jmcauley.ucsd.edu/data/amazon/ """
    df = get_pandas_df(path=path,
                       use_columns=['reviewerID', 'asin', 'unixReviewTime'],
                       n_rows=None)
    if min_item_per_user is not None:
        min_item_filter = df.groupby('reviewerID')['asin'].transform('count').ge(min_item_per_user)
        df = df[min_item_filter]

    df = df.sort_values(['unixReviewTime']).drop('unixReviewTime', axis=1)

    return df


def read_bert4rec_text_data(path):
    """ Reads data from a text file as used by Bert4Rec code. See https://github.com/FeiSun/BERT4Rec """
    df = pd.read_csv(path, delimiter=' ', header=None, dtype={0: str, 1: str})
    df = df.rename({0: 'reviewerID', 1: 'asin'}, axis=1)
    return df


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    MIN_ITEM = 5

    output_path = '../data/amazon_beauty_bert4rec'
    import shutil
    shutil.rmtree(output_path, ignore_errors=True)

    print('Reading data...')
    if True:
        # df = read_raw_amazon_data('../raw_data/reviews_Beauty.json.gz', min_item_per_user=MIN_ITEM)
        df = read_bert4rec_text_data('../raw_data/beauty.txt')
        df['side_1'] = np.random.uniform(size=len(df))
        df['side_2'] = np.random.uniform(size=len(df))

        print('# of interactions: ', len(df))

        item_vocab = pd.unique(df['asin'])
        print('# of items: ', len(item_vocab))

        n_users = len(pd.unique(df['reviewerID']))
        print('# of users: ', n_users)

        vocab_path = os.path.join(output_path, 'vocabs')
        os.makedirs(vocab_path, exist_ok=True)
        with open(os.path.join(vocab_path, 'item_vocab.txt'), 'w') as f:
            f.writelines('\n'.join(item_vocab))

        print('Converting to TF Examples...')

        tf_data = data_utils.pandas_to_tf_example_list(df, group_id_column='reviewerID')
        print('# of sequences: ', len(tf_data))

        print('Writing TFRecord files...')
        data_utils.write_to_tfrecord(tf_data, path=output_path, filename_prefix='amazon_beauty')
    else:
        # reading it back
        files = [os.path.join(output_path, filename) for filename in tf.io.gfile.listdir(output_path)]
        feature_spec = {
            'reviewerID': tf.io.FixedLenFeature([], tf.string),
            'asin': tf.io.VarLenFeature(tf.string),
            # 'unixReviewTime': tf.io.VarLenFeature(tf.int64)
        }

        dataset = tf.data.TFRecordDataset(files)

        for x in dataset.take(1):
            ex = tf.io.parse_single_example(x, feature_spec)
            # A VarLenFeature is always parsed to a SparseTensor
            for key in ex.keys():
                if isinstance(ex[key], tf.SparseTensor):
                    ex[key] = tf.sparse.to_dense(ex[key])

            print(ex)
