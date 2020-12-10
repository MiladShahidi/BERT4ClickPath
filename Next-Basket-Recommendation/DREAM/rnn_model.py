# -*- coding:utf-8 -*-

import torch
from torch.autograd import Variable
# from utils import data_helpers as dh
# import torch.nn.functional as F
from config import Config
#
# import os
# import math
# import time
# import logging
# import numpy as np
# from math import ceil
from utils import data_helpers as dh
# from torch.nn.functional import mse_loss
import warnings

warnings.filterwarnings("ignore")


class DRModel(torch.nn.Module):
    """
    Input Data: b_1, ... b_i ..., b_t
                b_i stands for user u's ith basket
                b_i = [p_1,..p_j...,p_n]
                p_j stands for the  jth product in user u's ith basket
    """

    def __init__(self, config):
        super(DRModel, self).__init__()

        # Model configuration
        # self.config = Config.from_dict(kwargs)
        self.config = config
        # config = self.config

        # Layer definitions
        # Item embedding layer, item's index

        # CORRECTION TODO: do not pad_idx
        self.encode = torch.nn.Embedding(num_embeddings=config.num_product,
                                         embedding_dim=config.embedding_dim)
        # self.encode = torch.nn.Embedding(num_embeddings=config.num_product,
        #                                  embedding_dim=config.embedding_dim,
        #                                  padding_idx=0)
        self.pool = {'avg': dh.pool_avg, 'max': dh.pool_max}[config.basket_pool_type]  # Pooling of basket

        # RNN type specify
        if config.rnn_type in ['LSTM', 'GRU']:
            if config.rnn_layer_num == 1:
                self.rnn = getattr(torch.nn, config.rnn_type)(input_size=config.embedding_dim,
                                                              hidden_size=config.embedding_dim,
                                                              num_layers=config.rnn_layer_num,
                                                              batch_first=True,
                                                              dropout=0.0,
                                                              bidirectional=False)
            else:
                self.rnn = getattr(torch.nn, config.rnn_type)(input_size=config.embedding_dim,
                                                              hidden_size=config.embedding_dim,
                                                              num_layers=config.rnn_layer_num,
                                                              batch_first=True,
                                                              dropout=config.dropout,
                                                              bidirectional=False)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[config.rnn_type]
            self.rnn = torch.nn.RNN(input_size=config.embedding_dim,
                                    hidden_size=config.embedding_dim,
                                    num_layers=config.rnn_layer_num,
                                    nonlinearity=nonlinearity,
                                    batch_first=True,
                                    dropout=config.dropout,
                                    bidirectional=False)
        # fully connected layer to connect the output of the LSTM cell to the output
        self.fc = torch.nn.Linear(in_features=config.num_product, out_features=config.num_product)
        self.softmax = torch.nn.Softmax(dim=2)
        self.relu = torch.nn.ReLU()
        # self.sig = torch.nn.(dim=1)
        # self.sigmoid = torch.nn.functional.sigmoid()

    def forward(self, x, lengths, dollar_amounts, hidden):
        # Basket Encoding
        # users' basket sequence
        ub_seqs = torch.Tensor(self.config.batch_size, self.config.seq_len, self.config.embedding_dim)
        for i, (baskets, dollars) in enumerate(
                zip(x, dollar_amounts)):  # shape of x: [batch_size, seq_len, indices of product]
            embed_baskets = torch.Tensor(self.config.seq_len, self.config.embedding_dim)
            for j, (basket, dollar) in enumerate(
                    zip(baskets, dollars)):  # shape of baskets: [seq_len, indices of product]
                # print('1')
                # print(basket)
                # print(type(basket), basket.shape)
                basket = torch.LongTensor(basket).resize_(1, len(basket))
                # print('2')
                # print(basket)
                # print(type(basket), basket.shape)
                basket = self.encode(torch.autograd.Variable(
                    basket))  # shape: [1, len(basket), embedding_dim] eg: torch.Size([1, 48, 5])
                # print('3')
                # print(basket)
                # print(type(basket), basket.shape)
                # exit()
                # print('dollar')

                # print('4')
                # print(dollar)
                # print(type(dollar), dollar.shape)


                # weigthed full basket with dollar amounts
                dollar = torch.FloatTensor(dollar)
                # print(dollar)
                # print(type(dollar), dollar.shape)
                dollar = dollar.view(1, -1, 1)
                # print(dollar)
                # print(type(dollar), dollar.shape)
                dollar_expanded = dollar.expand_as(basket)
                # print(dollar)
                # print(type(dollar), dollar.shape)
                # print('dollar end')
                basket = basket * dollar_expanded

                # print(basket)
                # print(type(basket), basket.shape)
                # exit()

                basket = self.pool(basket, dim=1)
                basket = basket.reshape(self.config.embedding_dim)
                embed_baskets[j] = basket  # shape:  [seq_len, 1, embedding_dim]
            # Concat current user's all baskets and append it to users' basket sequence
            ub_seqs[i] = embed_baskets  # shape: [batch_size, seq_len, embedding_dim]

        if self.config.varying_length_seq:
            # Packed sequence as required by pytorch is sequences have various length
            packed_ub_seqs = torch.nn.utils.rnn.pack_padded_sequence(ub_seqs, lengths, batch_first=True)
            output, h_u = self.rnn(packed_ub_seqs, hidden)
            # shape: [batch_size, true_len(before padding), embedding_dim]
            dynamic_user, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            packed_ub_seqs = ub_seqs
            # RNN
            output, h_u = self.rnn(packed_ub_seqs, hidden)
            dynamic_user = output

        # dynamic_user = self.softmax(dynamic_user)

        item_embedding = self.normalize(self.encode.weight)
        # item_embedding = self.encode.weight
        du_p_product_seqs = torch.Tensor(self.config.batch_size, self.config.seq_len, self.config.num_product)
        for i, du in enumerate(dynamic_user):  # shape of x: [batch_size, seq_len, indices of product]
            du_p_product = torch.mm(du, item_embedding.t())  # shape: [pad_len, num_item]
            du_p_product_seqs[i] = du_p_product  # shape: [batch_size, seq_len, embedding_dim]
        # du_p_product_seqs = dynamic_user

        # CORRECTION: TODO: add these two layer to the forward pass
        # du_p_product_seqs = self.fc(du_p_product_seqs)
        du_p_product_seqs = self.softmax(du_p_product_seqs)
        # du_p_product_seqs = self.relu(du_p_product_seqs)
        # for du_p_product_seq in du_p_product_seqs:
        #     du_p_product_seq = self.normalize(du_p_product_seq)
        return du_p_product_seqs

    def normalize(self, x):
        norm = x.abs().sum(1).view(-1, 1)
        norm[norm == 0] = 1.0
        x_normed = x / norm
        return x_normed

    def init_weight(self):
        # Init item embedding
        initrange = 0.3
        self.encode.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        # Init hidden states for rnn
        weight = next(self.parameters()).data
        if self.config.rnn_type == 'LSTM':
            return (Variable(weight.new(self.config.rnn_layer_num, batch_size, self.config.embedding_dim).zero_()),
                    Variable(weight.new(self.config.rnn_layer_num, batch_size, self.config.embedding_dim).zero_()))
        else:
            return Variable(torch.zeros(self.config.rnn_layer_num, batch_size, self.config.embedding_dim))

# model = DRModel(Config())


# print(model.encode(torch.LongTensor([0])))
# print(model.encode(torch.LongTensor([1])))
# # print(model.encode(torch.LongTensor([2])))
# model.encode(torch.autograd.Variable(
#                     basket))
# embedding(torch.LongTensor([1]))
