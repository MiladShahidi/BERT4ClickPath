# -*- coding:utf-8 -*-
__author__ = 'Fatemeh Renani'
class Config(object):
    def __init__(self):
        self.TRAININGSET_DIR = '../data/Train.json'
        self.VALIDATIONSET_DIR = '../data/Validation.json'
        self.TESTSET_DIR = '../data/Test.json'
        self.MODEL_DIR = 'runs/'
        self.cuda = False
        self.clip = 1
        self.epochs = 200

        self.batch_size = 1
        self.seq_len = 2
        self.learning_rate = 0.01  # Initial Learning Rate
        self.log_interval = 1  # num of batches between two logging
        self.basket_pool_type = 'avg'  # ['avg', 'max']
        self.rnn_type = 'LSTM'  # ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']
        self.rnn_layer_num = 2
        # TODO: have dropout
        self.dropout = 0.0
        self.num_product = 2  # Embedding Layer
        self.embedding_dim = 3  # Embedding Layer
        # self.neg_num = 500
        self.top_k = 10  # Top K
        self.varying_length_seq = False

        # TODO added for the refactoring purpose
        self.num_batches = 0
    @classmethod
    def from_dict(cls, param_dict):
        populated_instance = cls()
        for k in param_dict:
            setattr(populated_instance, k, param_dict[k])
        return populated_instance