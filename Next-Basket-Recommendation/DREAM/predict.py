# -*- coding:utf-8 -*-
__author__ = 'Fatemeh Renani'

import math
import logging
import torch
import pandas as pd
from utils import data_helpers as dh
import warnings
warnings.filterwarnings("ignore")

logging.info("DREAM Model Prediction...")

logger = dh.logger_fn("torch-log", "logs/is_training.log")

dilim = '-' * 120


def batch_predict(model, data, configuration):
    model.eval()
    dr_hidden = model.init_hidden(configuration.batch_size)

    full_basket = []
    for i in range(configuration.num_product):
        full_basket.append(i)
    full_basket = torch.LongTensor(full_basket)
    prediction_future = []
    count = 0

    num_batches = 0
    while num_batches <= configuration.num_batches:
        num_batches += 1
        uids, real_baskets, lens, dollars, baskets = next(data.generate_test_data())
        predicted_dollars = model(baskets, lens, dollars, dr_hidden)
        for pred_dollar, uid, real_basket, dollar, basket in zip(predicted_dollars, uids, real_baskets, dollars, baskets):
            pred_dollar_b = list(pred_dollar[configuration.seq_len - 1][full_basket].data.numpy())
            pred_dollar_b_copy = pred_dollar_b.copy()
            index_k = []
            for k in range(configuration.top_k):
                index = pred_dollar_b_copy.index(max(pred_dollar_b_copy))
                if index > 1e-6:
                    index_k.append(index)
                    pred_dollar_b_copy[index] = -9999
                else:
                    break
            if math.isnan(pred_dollar_b[0]):
                count += 1
            else:
                row = [uid, index_k, pred_dollar_b]
                prediction_future.append(row)
    print(prediction_future)
    # prediction_future = pd.DataFrame(prediction_future, columns=['userID', 'pred_basket', 'pred_dollar_amount'])
    # prediction_future.to_json('prediction_future.json', orient='record', lines=True)
    # logger.info(count)
