# -*- coding:utf-8 -*-
__author__ = ''

import os
import math
import time
import logging
import torch
import numpy as np
import pandas as pd
from math import ceil
from utils import data_helpers as dh
from utils.data_generator import ReturnsDataGen
# from config import Config
from rnn_model import DRModel
from torch.nn.functional import mse_loss
import warnings

warnings.filterwarnings("ignore")

logging.info('torch version in used', torch.__version__)
logging.info("DREAM Model Training...")

logger = dh.logger_fn("torch-log", "logs/is_training.log")

# dilim = '-' * 120
# logger.info(dilim)
# for attr in sorted(Config().__dict__):
#     logger.info('{:>50}|{:<50}'.format(attr.upper(), Config().__dict__[attr]))
# logger.info(dilim)


def recall_precision(top_k, real_basket, pred_basket, recall, precision):
    '''
    This function calculate the recall and precision metrics for the top top_k item in the basket.
    def: Recall@k measures the number of correct items predicted by the model divided by the basket size.
    Recall@k = (# of predicted items @k that are correct) / (total # of items in real basket)

    :param top_k: int indicating the number of k top items to consider
    :param real_basket: list(int) list of items in the real basket
    :param pred_basket: list(float) list of items' dollar amount in the predicted basket
    :return:
    :param recall: float
    :param precision: float
    '''

    index_k = []
    # print(pred_basket)
    for k in range(top_k):
        # CORRECTION TODO: add the max_value and have the next if condiction on the value instead of index
        max_value = max(pred_basket)
        index = pred_basket.index(max_value)
        if max_value > 1e-6:
            index_k.append(index)
            pred_basket[index] = -9999
        else:
            break
    # print('@', real_basket, index_k)

    # Calculate number of correctly predicted items
    u_num_corect_pred = 0
    for k in range(len(index_k)):
        if index_k[k] in real_basket:
            u_num_corect_pred += 1
    # CORRECTION TODO: change len(real_basket) to len(real_basket[0])
    recall += u_num_corect_pred / len(real_basket[0])
    if len(index_k) != 0:
        precision += u_num_corect_pred / len(index_k)
    # print(real_basket, index_k, u_num_corect_pred, recall, precision, len(real_basket), len(real_basket[0]))
    return recall, precision


def loss_fun(pred_dollars, dollars, real_baskets, configuration):
    '''
    :param pred_dollars: batch of users' predicted dollars
    :param dollars: batch of users' real dollar amounts
    :param real_baskets: list(int) batch of users' real baskets
    :return:
    '''

    loss = 0.0
    recall = 0.0
    precision = 0.0
    full_basket = []
    for i in range(0, configuration.num_product):
        full_basket.append(i)
    full_basket = torch.LongTensor(full_basket)
    for pred_dollar, dollar, real_basket in zip(pred_dollars, dollars, real_baskets):
        loss_u = []  # loss for user
        for t in range(1, configuration.seq_len):
            loss_mse = mse_loss(pred_dollar[t - 1][full_basket], torch.FloatTensor(dollar[t]), size_average=False)
            loss_u.append(loss_mse)
            if t == configuration.seq_len - 1:  # for the last basket calculate recall, precision
                pred_basket = list(pred_dollar[t - 1][full_basket].data.numpy())
                # print('1$$$$$$$$$$$$')
                # print('*', real_basket)
                # # print('*', pred_dollar[t - 1][full_basket].data.numpy())
                # print('*',  dollar)
                # print("*", pred_dollar)
                # print('2$$$$$$$$$$$$')
                recall, precision = recall_precision(configuration.top_k, real_basket[t], pred_basket, recall, precision)
        for i in loss_u:
            loss = loss + i / len(loss_u)

    avg_loss = torch.div(loss, len(pred_dollars))
    avg_recall = torch.div(recall, len(pred_dollars))
    avg_precision = torch.div(precision, len(pred_dollars))
    if (recall + precision) != 0.0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0.0

    avg_f1 = torch.div(f1, len(pred_dollars))
    return avg_loss, avg_recall, avg_precision, avg_f1


def train_model(model, data,  optimizer, epoch, configuration):
    model.train()  # turn on is_training mode for dropout
    # configuration = Config()
    dr_hidden = model.init_hidden(configuration.batch_size)
    train_loss = 0
    train_recall = 0
    train_precision = 0
    train_f1 = 0
    start_time = time.perf_counter()
    num_batches = 0

    while num_batches <= configuration.num_batches:
        num_batches += 1

        uids, real_baskets, lens, dollars, baskets = next(data.generate_train_data())
        # print(dollars)
        # print(real_baskets)
        # print('*' * 10)
        model.zero_grad()  #
        pred_dollars = model(baskets, lens, dollars, dr_hidden)


        # print('*' * 89)
        # print(dollars)
        # print()
        # print(pred_dollars)
        # print('*'*89)

        loss, recall, precision, f1 = loss_fun(pred_dollars, dollars, real_baskets, configuration)
        loss.backward()
        # Clip to avoid gradient exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), configuration.clip)

        # Parameter updating
        optimizer.step()
        train_loss += loss.data
        train_recall += recall
        train_precision += precision
        train_f1 += f1

        # Logging
        if num_batches % configuration.log_interval == 0:
            elapsed = (time.perf_counter() - start_time) / configuration.log_interval
            cur_loss = train_loss.item() / configuration.log_interval  # turn tensor into float
            cur_recall = train_recall / configuration.log_interval
            cur_precision = train_precision / configuration.log_interval
            cur_f1 = train_f1 / configuration.log_interval
            train_loss = 0
            train_recall = 0
            train_precision = 0
            train_f1 = 0
            start_time = time.perf_counter()
            logger.info('[Training]| Epochs {:3d} | Batch {:5d} / {:5d} | ms/batch {:02.2f} | Loss {:05.4f} | '
                        'recall {:05.4f} | percision {:05.4f} | f1 {:05.4f} |'.format(epoch, num_batches, configuration.num_batches,
                                                                                      elapsed, cur_loss, cur_recall,
                                                                                      cur_precision, cur_f1))
    return model


def validate_model(model, data, epoch, configuration):
    model.eval()
    dr_hidden = model.init_hidden(configuration.batch_size)

    # calculating the loss, recall, precision for the validation dataset in the eva mode
    val_loss = 0.0
    val_recall = 0.0
    val_precision = 0.0
    val_f1 = 0.0
    start_time = time.perf_counter()
    num_batches = 0 #ceil(len(validation_data) / Config().batch_size)
    # for i, x in enumerate(dh.batch_iter(validation_data, Config().batch_size, Config().seq_len, shuffle=False)):
    while num_batches <= configuration.num_batches:
        num_batches += 1
        uids, real_baskets, lens, dollars, baskets = next(data.generate_validation_data())
        predicted_dollars = model(baskets, lens, dollars, dr_hidden)

        print('#' * 89)
        print(dollars)
        print()
        print(predicted_dollars)
        print('#' * 89)


        loss, recall, precision, f1 = loss_fun(predicted_dollars, dollars, real_baskets, configuration)
        val_loss += loss.data
        val_recall += recall
        val_precision += precision
        val_f1 += f1
    # Logging
    elapsed = (time.perf_counter() - start_time) * 1000 / num_batches
    val_loss = val_loss.item() / num_batches
    val_recall = val_recall / num_batches
    val_precision = val_precision / num_batches
    val_f1 = val_f1 / num_batches
    logger.info('[Validation]| Epochs {:3d} | Elapsed {:02.2f} | Loss {:05.4f} | recall {:05.4f} |'
                'precision {:05.4f} |f1 {:05.4f} |'.format(epoch, elapsed, val_loss, val_recall, val_precision,
                                                           val_f1))
    return val_loss, val_recall, val_precision, val_f1


def test_model(model, data,  optimizer, epoch, configuration):#(model, train_data, test_data, epoch):
    # MODEL_DIR = 'model/best_model.model'
    # model = torch.load(MODEL_DIR)
    model.eval()
    dr_hidden = model.init_hidden(Config().batch_size)
    test_loss = 0
    test_recall = 0
    test_precision = 0
    test_f1 = 0
    full_basket = []
    for j in range(0, Config().num_product):
        full_basket.append(j)
    full_basket = torch.LongTensor(full_basket)
    num_batches = ceil(len(train_data) / Config().batch_size)
    for i, x in enumerate(dh.batch_iter(train_data, Config().batch_size, Config().seq_len, shuffle=False)):
        uids, real_baskets, lens, dollars, baskets = x
        pred_dollars = model(baskets, lens, dollars, dr_hidden)
        score = 0
        recall = 0
        precision = 0
        f1 = 0
        for dollar, pred_dollar, uid in zip(pred_dollars, dollars, uids):
            real_test_basket = test_data[test_data['userID'] == uid].baskets.values[0]  # list dim 1
            test_b_dollar = test_data[test_data['userID'] == uid].dollar_amount.values[0]

            pred_dollar = torch.FloatTensor(pred_dollar)
            score += mse_loss(pred_dollar[Config().seq_len - 1][full_basket], torch.FloatTensor(test_b_dollar),
                              size_average=False)
            pred_test_basket = list(pred_dollar[Config().seq_len - 1][full_basket].data.numpy()).copy()
            recall, precision = recall_precision(Config().top_k, real_test_basket, pred_test_basket, recall, precision)

            if (recall + precision) != 0.0:
                f1 = 2 * recall * precision / (recall + precision)
            else:
                f1 = 0.0
            if math.isnan(score):
                print('Warning::Nan was predicted for the test basket')
        test_loss += score.data / Config().batch_size
        test_recall += recall / Config().batch_size
        test_precision += precision / Config().batch_size
        test_f1 += f1 / Config().batch_size

        print_status = False  # if True the last predicted basket and its corresponding dollar amount will be printed
        if print_status:
            last_b_dollar = list(pred_dollars[Config().seq_len - 1][full_basket].data.numpy()).copy()
            index_k = []
            for k in range(Config().top_k):
                index = last_b_dollar.index(max(last_b_dollar))
                if index > 1e-6:
                    index_k.append(index)
                    last_b_dollar[index] = -9999
                else:
                    break
            logger.info(uid)
            logger.info(real_test_basket)
            logger.info(index_k)

            last_b_dollar = test_b_dollar.copy()
            index_k = []
            for k in range(Config().top_k):
                index = last_b_dollar.index(max(last_b_dollar))
                if index > 1e-6:
                    index_k.append(index)
                    last_b_dollar[index] = -9999
                else:
                    break
            logger.info(index_k)
            logger.info(-np.sort(-pred_dollars[Config().seq_len - 1][full_basket].data.numpy())[:10])
            logger.info(-np.sort(-np.array(test_b_dollar))[:10])
            logger.info(test_b_dollar)
            logger.info('     ')

    logger.info('[Test]| Epochs {:3d} | loss {:05.4f} | recall {:05.4f} | precision {:05.4f} | f1 {:05.4f} |'
                .format(epoch, test_loss / num_batches, test_recall / num_batches, test_precision / num_batches,
                        test_f1 / num_batches))
    return


def train():

    # Model config
    model = DRModel()
    data = ReturnsDataGen(Config().batch_size, Config().num_product, Config().seq_len)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config().learning_rate)

    timestamp = str(int(time.time()))

    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger.info('Save into {0}'.format(out_dir))
    checkpoint_dir = out_dir + '/model-{epoch:02d}.model'
    best_val_loss = None

    try:
        # Training
        for epoch in range(Config().epochs):
            model = train_model(model, data, optimizer, epoch)
            logger.info('-' * 89)
            #
            # val_loss = validate_model(model, train_data, validation_data, epoch)
            # logger.info('-' * 89)
            #
            # test_model(model, train_data, test_data, epoch)
            # logger.info('-' * 89)

            # if not best_val_loss or val_loss < best_val_loss:
            #     with open(checkpoint_dir.format(epoch=epoch, val_loss=val_loss), 'wb') as f:
            #         torch.save(model, f)
            #     best_val_loss = val_loss
            with open(checkpoint_dir.format(epoch=epoch), 'wb') as f:
                torch.save(model, f)
    except KeyboardInterrupt:
        logger.info('*' * 89)
        logger.info('Early Stopping!')


if __name__ == '__main__':
    train()