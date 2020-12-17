# -*- coding:utf-8 -*-
__author__ = ''

import os
import logging
import torch
import numpy as np
import pandas as pd


def logger_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger


def load_data(input_file, flag=None):
    if flag:
        data = pd.read_json(input_file, orient='records', lines=True)
    else:
        data = pd.read_json(input_file, orient='records', lines=True)

    return data


def load_model_file(checkpoint_dir):
    MODEL_DIR = 'runs/' + checkpoint_dir
    names = [name for name in os.listdir(MODEL_DIR) if os.path.isfile(os.path.join(MODEL_DIR, name))]
    max_epoch = 0
    choose_model = ''
    for name in names:
        if int(name[6:8]) >= max_epoch:
            max_epoch = int(name[6:8])
            choose_model = name
    MODEL_FILE = 'runs/' + checkpoint_dir + '/' + choose_model
    return MODEL_FILE


def sort_batch_of_lists(uids, batch_of_baskets, lens, batch_of_dollar, batch_of_full_baskets):
    """Sort batch of lists according to len(list). Descending"""
    sorted_idx = [i[0] for i in sorted(enumerate(lens), key=lambda x: x[1], reverse=True)]
    uids = [uids[i] for i in sorted_idx]
    lens = [lens[i] for i in sorted_idx]
    batch_of_baskets = [batch_of_baskets[i] for i in sorted_idx]
    batch_of_dollar = [batch_of_dollar[i] for i in sorted_idx]
    batch_of_full_baskets = [batch_of_full_baskets[i] for i in sorted_idx]
    return uids, batch_of_baskets, lens, batch_of_dollar, batch_of_full_baskets


def pad_batch_of_lists(batch_of_lists, pad_len):
    """Pad batch of lists."""
    padded = [l + [[0]] * (pad_len - len(l)) for l in batch_of_lists]
    return padded


def batch_iter(batch_size, n_items, seq_len):
    """
    Turn dataset into iterable batches.

    Args:
        batch_size: The size of the data batch
        num_product: The Number of products
    Returns:
        A batch iterator for data set
    """

    ## TO-DO: when the generate generate a seq of basket with the same size then the shape of baskets[i] is (seq_len, 1, length_basket) in oppos to the other case where the shape of
    # basktes[i] is (seq_len, 1). So there is a error such as "ValueError: could not broadcast input array from shape (4,1,7) into shape (4,1)"
    n_user = 1000
    uids = np.random.choice(range(n_user), batch_size, replace=False)
    baskets = []
    dollar = []
    lens = []
    full_baskets = []
    # full_basket = np.array(range(n_items))
    for user in uids:
        u_baskets = []
        u_dollar_amounts = []
        u_lens = []
        u_full_baskets = []
        for i in range(seq_len):
            basket_length = np.random.randint(1, n_items)
            # u_baskets.append(np.random.choice(range(n_items), basket_length, replace=False))
            # u_dollar_amounts.append(np.random.rand(basket_length))
            u_dollar_help = np.random.rand(n_items)
            max_zero = np.random.randint(0, n_items)
            for j in range(max_zero):
                index = np.random.randint(0, n_items)
                u_dollar_help[index] = 0.0
            u_baskets.append(np.nonzero(u_dollar_help))
            u_dollar_amounts.append(u_dollar_help)
            # u_full_baskets.append(np.array(range(n_items)))
            u_full_baskets.append(np.array(range(n_items)))

        u_baskets = np.array(u_baskets)
        baskets.append(u_baskets)
        u_dollar_amounts = np.array(u_dollar_amounts)
        dollar.append(u_dollar_amounts)
        # lens.append(u_lens)
        lens.append(u_baskets.shape[0])
        u_full_baskets = np.array(u_full_baskets)
        full_baskets.append(u_full_baskets)

    print(baskets[0])
    print('shape', (len(baskets)), (baskets[0].shape), dollar[0].shape)
    baskets = np.array(baskets)
    dollar = np.array(dollar)
    full_baskets = np.array(full_baskets)
    lens = np.array(lens)

    uids, baskets, lens, dollar, full_baskets = sort_batch_of_lists(uids, baskets, lens, dollar, full_baskets)  #
    yield uids, baskets, lens, dollar, full_baskets


def inference_iter(data, batch_size, pad_len, shuffle=True):
    """
    Turn dataset into iterable batches of 1 for inference.

    Args:
        data: The data
        batch_size: The size of the data batch
        pad_len: The padding length
        shuffle: Shuffle or not (default: True)
    Returns:
        A batch iterator for data set
    """
    data_size = len(data)
    num_batches_per_epoch = data_size
    if shuffle:
        shuffled_data = data.sample(frac=1)
    else:
        shuffled_data = data

    for i in range(num_batches_per_epoch):
        uids = shuffled_data[i * batch_size: (i + 1) * batch_size].userID.values
        baskets = list(shuffled_data[i * batch_size: (i + 1) * batch_size].baskets.values)
        lens = shuffled_data[i * batch_size: (i + 1) * batch_size].num_baskets.values
        dollar = shuffled_data[i * batch_size: (i + 1) * batch_size].dollar_amount.values
        full_baskets = list(shuffled_data[i * batch_size: (i + 1) * batch_size].full_baskets.values)

        uids, baskets, lens, dollar, full_baskets = sort_batch_of_lists(uids, baskets, lens, dollar, full_baskets)  #
        yield uids, baskets, lens, dollar, full_baskets


def pool_max(tensor, dim):
    return torch.max(tensor, dim)[0]


def pool_avg(tensor, dim):
    return torch.mean(tensor, dim)