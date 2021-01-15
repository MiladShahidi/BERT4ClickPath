import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sequence_transformer.data_prep import data_utils
from applications.NBR.NBR_constant import SEQ_LEN, MIN_SEQ_LEN


def process_raw_data(filename, columns_name, pickle_file, limit_user_item=10):
    """
    Read transactional data and create a dataframe that has list of basket for each user
    Args:
        filename: path (including name of the csv file) to the transaction data in csv format
        columns_name: List of original column names in the transaction dataset which correspond to
                        userID, itemID, time_of_transaction
        pickle_file: path (including pickle name) to save dictionary of item vocabulary
        limit_user_item: use to eliminate items that have been purchased by less that this value and also
                            eliminate the user that has purchased less than

    Returns:
        baskets: pandas datafram with two columns named usrID and basket
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

    # # convert PRODUCT_ID to categorical variable
    # df['itemID'] = df['itemID'].astype('category')
    # dic_product_id = {k + 1: v for k, v in enumerate(df['itemID'].cat.categories)}
    # df['itemID'] = df['itemID'].cat.codes + 1  # start the item index from 1
    # # with open(pickle_file, 'wb') as handle:
    # #     pickle.dump(dic_product_id, handle)

    df['datetime'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('datetime')

    df = df.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    df = df[['userID', 'itemID']]
    df.columns = ['userID', 'basket']

    baskets = df.groupby(['userID'])['basket'].apply(list).reset_index()
    # baskets.set_index('userID', inplace=True)

    return baskets


def train_test_val_split_vocab(df,  vocab_path, vocab_name):
    """
    Create test and train and validation dataset for next basket recommendation
    Example:
    Given the following Pandas DataFrame:

           userID                        basket                     len
    0       1  [[101], [302], [498], [197], [175], [147], [194]]    7

    if SEQ_LEN >= 7 (number of baskets)
        the The test df would be:
                              feature  label
        userID
        1       [[101], [302], [498], [197], [175], [147]]  [194]

        The train dataset is
                       feature  label
        userID
        1       [[101], [302], [498], [197], [175]]  [147]

    if SEQ_LEN = 4 (which is les than the number of baskets )
        the The test df would be:
                              feature  label
        userID
        1       [ [197], [175], [147]]  [194]

        The train dataset is
                       feature  label
        userID
        1       [[498], [197], [175]]  [147]

    Args:
        df: A Pandas DataFrame.

    Returns:
        test_df: A Pandas DataFrame
        train_df: A Pandas DataFrame
        ev_df: A Pandas DataFrame
    """
    def dataset_preparation(df):
        # For the user who has same or more baskets than SEQ_LEN, the most the most recent baskets are selected
        # and the last basket is set to label
        df['len'] = df['basket'].apply(len)
        df1 = df.copy()
        df1['feature']= df[df['len']>= SEQ_LEN]['basket'].apply(lambda x: x[-SEQ_LEN:-1])
        df1['label']= df[df['len']>= SEQ_LEN]['basket'].apply(lambda x: x[-1])
        df1.dropna(inplace=True)
        df1 = df1[['userID','feature', 'label']]
        # print(df1)

        # For the user who has les baskets than SEQ_LEN the last basket is set to label
        df2 = df.copy()
        # df2 = df['userID']
        df2['feature']= df[df['len']< SEQ_LEN]['basket'].apply(lambda x: x[0:-1])
        df2['label']= df[df['len'] < SEQ_LEN]['basket'].apply(lambda x: x[-1])
        df2.dropna(inplace=True)
        # print(df2)
        df2 = df2[['userID', 'feature', 'label']]

        data = pd.concat([df2, df1])#.set_index('userID')
        del df1, df2
        data.set_index('userID', inplace=True)
        return data

    def create_vocabulary(df, path, filename):
        """
        convert itemID into range of [1-size-of-vocabulary]
        create and write a vocabulary of item in training dataset and return vocabulary size
        Args:
            df:
            path:
            filename:

        Returns:
            vocab_len : vocabulary size
            dic_product_id : dictionary of {original item-id: converted item-id}
        """
        import itertools
        df['all_items'] = df.basket.apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))

        vocab = []
        for value in df.all_items:
            vocab.extend(value)
        vocab = list(set(vocab))
        vocab.sort()

        dic_product_id = {item: str(i+1) for i, item in enumerate(vocab)}
        # add unkown token to the dictionary
        # dic_product_id['unknown'] = len(vocab) + 1

        vocab = [i for i in range(1, len(vocab)+1)]
        # vocab += ['unknown']

        # drop the intermadiate column 'all_items
        df.drop(['all_items'], axis=1, inplace=True)

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            filelist = [f for f in os.listdir(path) if f.endswith(".txt")]
            for f in filelist:
                os.remove(os.path.join(path, f))

        with open(os.path.join(path, filename), mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(str(line) for line in vocab))

        return len(vocab), dic_product_id

    def mark_training_samples(df, train_size):
        indices = np.random.choice(df.index, size=train_size, replace=False)
        train_indices = df.index.isin(indices)
        df['train'] = train_indices
        return df

    def convert_item_id_correct_range(baskets, dic):
        if isinstance(baskets[0], list):  # Examine one value to see whether this is a list or a list of lists
            converted_bs = []
            for basket in baskets:
                converted_b = []
                for item in basket:
                    if item in dic.keys():
                        converted_b.append(dic[item])
                    # else:
                    #     converted_b.append(dic['unknown'])
                converted_bs.append(converted_b)
            return converted_bs
        else:
            converted_b = []
            for item in baskets:
                if item in dic.keys():
                    converted_b.append(dic[item])
                # else:
                #     converted_b.append(dic['unknown'])
            return converted_b

    # create the test dataset by selecting the last basket as label
    test_df = dataset_preparation(df)

    # remove the last basket which is reserved for test dataset and process the train+validation dataset
    df['basket'] = df['basket'].apply(lambda x: x[0:-1])

    vocab_size, dic = create_vocabulary(df, vocab_path, vocab_name)
    df = dataset_preparation(df)
    df['feature'] = df.feature.apply(lambda x: convert_item_id_correct_range(x, dic))
    df['label'] = df.label.apply(lambda x: convert_item_id_correct_range(x, dic))

    test_df['feature'] = test_df.feature.apply(lambda x: convert_item_id_correct_range(x, dic))
    test_df['label'] = test_df.label.apply(lambda x: convert_item_id_correct_range(x, dic))

    df = mark_training_samples(df, int(df.shape[0]*0.8))
    train_df = df[df['train']].drop('train', axis=1)
    eval_df = df[~df['train']].drop('train', axis=1)

    def multi_hop_converter(items, size):
        one_hoted = np.zeros(size)
        for item in items:
            one_hoted[int(item) - 1] = 1.0
        return list(one_hoted)#list( map(float, one_hoted))

    # print(one_hop_converter(items))
    test_df['label'] = test_df.label.apply(lambda x: multi_hop_converter(x, size=vocab_size))
    train_df['label'] = train_df.label.apply(lambda x: multi_hop_converter(x, size=vocab_size))
    eval_df['label'] = eval_df.label.apply(lambda x: multi_hop_converter(x, size=vocab_size))
    return test_df, train_df, eval_df


if __name__ == '__main__':

    import datetime

    start = datetime.datetime.now()
    transaction_dataset = True
    sample_basket = [1,2,3,4]

    if transaction_dataset:
        df = process_raw_data('../data/raw_data/ta-fen-transaction.csv',
                              ['CUSTOMER_ID', 'PRODUCT_ID', 'TRANSACTION_DT'],
                              '../data/item_vocab.pickle', limit_user_item=10)
    else:
        df = pd.DataFrame({
            'userID': [1, 1, 1, 1, 1,1,1,1,1,1, 1, 2,2,2,2,2,2,2,2,2,2,2],
            # 'basket': [list(np.random.randint(100, 200, size=np.random.randint(low=1, high=2))) for _ in range(13)]
            'basket': [sample_basket for _ in range(22)]
        })
        df = df.groupby(['userID'])['basket'].apply(list).reset_index()

    print(df)
    # exit()
    df['len'] = df['basket'].apply(len)
    df = df[df['len'] >= MIN_SEQ_LEN]
    df.drop(columns=['len'])

    df['userID'] = df['userID'].astype('str')
    # print(df)
    vocab_path = '../data/vocabs'
    vocab_name = 'item_vocab.txt'
    test, train, validation = train_test_val_split_vocab(df,  vocab_path, vocab_name)

    # end = datetime.datetime.now()
    # # print('1', start-end)
    # start = datetime.datetime.now()

    print(test)
    print(train)
    print(validation)
    # exit()
    test = data_utils.pandas_to_seq_example(test, 'userID', ['feature'], ['label'])
    train = data_utils.pandas_to_seq_example(train, 'userID', ['feature'], ['label'])
    validation = data_utils.pandas_to_seq_example(validation, 'userID', ['feature'], ['label'])

    end = datetime.datetime.now()
    print('2', start - end)
    start = datetime.datetime.now()

    test_path = '../data/test/'
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    else:
        filelist = [f for f in os.listdir(test_path) if f.endswith(".tfrecord")]
        for f in filelist:
            os.remove(os.path.join(test_path, f))
        # os.remove(os.path.join(test_path, '*.tfrecord'))
    train_path = '../data/train/'
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    else:
        filelist = [f for f in os.listdir(train_path) if f.endswith(".tfrecord")]
        for f in filelist:
            os.remove(os.path.join(train_path, f))

    val_path = '../data/validation/'
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    else:
        filelist = [f for f in os.listdir(val_path) if f.endswith(".tfrecord")]
        for f in filelist:
            os.remove(os.path.join(val_path, f))

    end = datetime.datetime.now()
    print('3', start - end)
    start = datetime.datetime.now()

    data_utils.write_to_tfrecord(test, test_path, 'test_data')
    data_utils.write_to_tfrecord(train, train_path, 'train_data')
    data_utils.write_to_tfrecord(validation, val_path, 'validation_data')
    # Reading it back

    # tf_dataset = tf.data.TFRecordDataset('../data/test/test_data_1_of_2.tfrecord')
    tf_dataset = tf.data.TFRecordDataset('../data/test/test_data_0_of_1.tfrecord')

    context_feature_spec = {
        'userID': tf.io.FixedLenFeature([], tf.string),
        'feature1': tf.io.VarLenFeature(tf.string),
        'feature2': tf.io.VarLenFeature(tf.string),
        'feature3': tf.io.VarLenFeature(tf.string),
        'feature4': tf.io.VarLenFeature(tf.string),
        'feature5': tf.io.VarLenFeature(tf.string),
        'feature6': tf.io.VarLenFeature(tf.string),
        'feature7': tf.io.VarLenFeature(tf.string),
        'feature8': tf.io.VarLenFeature(tf.string),
        'feature9': tf.io.VarLenFeature(tf.string),
        'feature10': tf.io.VarLenFeature(tf.string),
        'label': tf.io.VarLenFeature(tf.float32)
    }

    end = datetime.datetime.now()
    print('4', start - end)
    start = datetime.datetime.now()
    # sequence_feature_spec = {
    #     'feature': tf.io.VarLenFeature(tf.int64)
    # }
    def parse_examples(x):
        features = tf.io.parse_example(serialized=x, features=context_feature_spec)
        # A VarLenFeature is always parsed to a SparseTensor
        for key in features.keys():
            if isinstance(features[key], tf.SparseTensor):
                features[key] = tf.sparse.to_dense(features[key])
        return features


    def parse_fn(x):
        parsed_context, parsed_sequence = tf.io.parse_single_sequence_example(
            serialized=x,
            context_features=context_feature_spec,
            # sequence_features=sequence_feature_spec
        )
        # A VarLenFeature is always parsed to a SparseTensor
        features_dict = parsed_context  # {**parsed_sequence, **parsed_context}
        for key in features_dict.keys():
            if isinstance(features_dict[key], tf.SparseTensor):
                features_dict[key] = tf.sparse.to_dense(features_dict[key])
        return features_dict


    end = datetime.datetime.now()
    print('5', start - end)
    start = datetime.datetime.now()

    tf_dataset = tf_dataset.map(parse_examples)

    INPUT_PAD = 0
    INPUT_PADDING_TOKEN = '[PAD]'

    print('INPUT_PAD', INPUT_PAD)

    tf_dataset = tf_dataset.padded_batch(
        batch_size=5,
        padded_shapes={  # Pad all to longest in batch
            'userID': [],
            'feature1': [None],
            'feature2': [None],
            'feature3': [None],
            'feature4': [None],
            'feature5': [None],
            'feature6': [None],
            'feature7': [None],
            'feature8': [None],
            'feature9': [None],
            'feature10': [None],
            'label': [None]
            # 'feature': [SEQ_LEN, None]
        },
        padding_values={
            'userID': tf.cast(INPUT_PADDING_TOKEN,   tf.string),
            'feature1': tf.cast(INPUT_PADDING_TOKEN, tf.string),
            'feature2': tf.cast(INPUT_PADDING_TOKEN, tf.string),
            'feature3': tf.cast(INPUT_PADDING_TOKEN, tf.string),
            'feature4': tf.cast(INPUT_PADDING_TOKEN, tf.string),
            'feature5': tf.cast(INPUT_PADDING_TOKEN, tf.string),
            'feature6': tf.cast(INPUT_PADDING_TOKEN, tf.string),
            'feature7': tf.cast(INPUT_PADDING_TOKEN, tf.string),
            'feature8': tf.cast(INPUT_PADDING_TOKEN, tf.string),
            'feature9': tf.cast(INPUT_PADDING_TOKEN, tf.string),
            'feature10': tf.cast(INPUT_PADDING_TOKEN, tf.string),
            'label': tf.cast(INPUT_PAD, tf.float32)
        }
    )
    from pprint import pprint

    end = datetime.datetime.now()
    print('6', start - end)
    start = datetime.datetime.now()

    for x in tf_dataset:
        pprint(x)

