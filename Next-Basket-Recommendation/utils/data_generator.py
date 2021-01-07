import numpy as np
from enum import Enum

class ReturnsDataGen:

    def __init__(self, batch_size, num_product, seq_len):

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_product = num_product
        self._n_user = 1000

    def sort_batch_of_lists(self, uids, batch_of_baskets, lens, batch_of_dollar, batch_of_full_baskets):
        """Sort batch of lists according to len(list). Descending"""
        sorted_idx = [i[0] for i in sorted(enumerate(lens), key=lambda x: x[1], reverse=True)]
        uids = [uids[i] for i in sorted_idx]
        lens = [lens[i] for i in sorted_idx]
        batch_of_baskets = [batch_of_baskets[i] for i in sorted_idx]
        batch_of_dollar = [batch_of_dollar[i] for i in sorted_idx]
        batch_of_full_baskets = [batch_of_full_baskets[i] for i in sorted_idx]
        return uids, batch_of_baskets, lens, batch_of_dollar, batch_of_full_baskets

    # def generate_data(self):
    #     """
    #     """
    #     uids = np.random.choice(range(self._n_user), self.batch_size, replace=False)
    #     baskets = []
    #     dollar = []
    #     lens = []
    #     full_baskets = []
    #     for user in uids:
    #         u_baskets = []
    #         u_dollar_amounts = []
    #         u_full_baskets = []
    #         for i in range(self.seq_len):
    #             # basket_length = np.random.randint(1, self._n_items)
    #             u_dollar_help = np.random.rand(self.num_product)
    #             u_dollar_help = u_dollar_help/u_dollar_help.sum(axis=0,keepdims=1)
    #             max_zero = np.random.randint(0, self.num_product)
    #             for j in range(max_zero):
    #                 index = np.random.randint(0, self.num_product)
    #                 u_dollar_help[index] = 0.0
    #             u_baskets.append(np.nonzero(u_dollar_help))
    #             u_dollar_amounts.append(u_dollar_help)
    #             u_full_baskets.append(np.array(range(self.num_product)))
    #
    #         baskets.append(u_baskets)
    #         u_dollar_amounts = np.array(u_dollar_amounts)
    #         dollar.append(u_dollar_amounts)
    #         lens.append(np.array(u_baskets).shape[0])
    #         u_full_baskets = np.array(u_full_baskets)
    #         full_baskets.append(u_full_baskets)
    #     baskets = np.array(baskets)
    #     dollar = np.array(dollar)
    #     full_baskets = np.array(full_baskets)
    #     lens = np.array(lens)
    #     uids, baskets, lens, dollar, full_baskets = self.sort_batch_of_lists(uids, baskets, lens, dollar, full_baskets)
    #     yield uids, baskets, lens, dollar, full_baskets

    def generate_data(self):
        """
        """
        uids = np.random.choice(range(self._n_user), self.batch_size, replace=False)
        baskets = []
        dollar = []
        lens = []
        full_baskets = []
        np.random.seed(0)

        for user in uids:
            u_baskets = []
            u_dollar_amounts = []
            u_full_baskets = []
            # u_dollar_same = np.random.rand(self.num_product)
            # u_dollar_same[0] = 0.0
            # u_dollar_same[1] = 0.0
            u_dollar_same = np.array([1,1,1,1,1,1,1,1,1,1])
            u_dollar_help = u_dollar_same.copy()
            # index = np.random.randint(0, self.num_product)
            # u_dollar_help[index] = 0.0

            # max_zero = np.random.randint(0, self.num_product)
            # for j in range(max_zero):
            #     index = np.random.randint(0, self.num_product)
            #     u_dollar_help[index] = 0.0
            u_dollar_help = u_dollar_help / u_dollar_help.sum(axis=0, keepdims=1)
            for i in range(self.seq_len):
                # basket_length = np.random.randint(1, self._n_items)
                # u_dollar_help = np.random.rand(self.num_product)

                u_baskets.append(np.nonzero(u_dollar_help))
                u_dollar_amounts.append(u_dollar_help)
                u_full_baskets.append(np.array(range(self.num_product)))

            baskets.append(u_baskets)
            u_dollar_amounts = np.array(u_dollar_amounts)
            dollar.append(u_dollar_amounts)
            lens.append(np.array(u_baskets).shape[0])
            u_full_baskets = np.array(u_full_baskets)
            full_baskets.append(u_full_baskets)
        baskets = np.array(baskets)
        dollar = np.array(dollar)
        full_baskets = np.array(full_baskets)
        lens = np.array(lens)
        uids, baskets, lens, dollar, full_baskets = self.sort_batch_of_lists(uids, baskets, lens, dollar, full_baskets)
        # print(dollar)
        # print(baskets)
        # print('*'*10)
        yield uids, baskets, lens, dollar, full_baskets

    def generate_train_data(self):
        return self.generate_data()

    def generate_validation_data(self):
        return self.generate_data()

    def generate_test_data(self):
        return self.generate_data()

    def get_generator(self, batch_data_type):
        mapping = {ModelDataBatchType.Training: self.generate_train_data,
                   ModelDataBatchType.Test: self.generate_test_data,
                   ModelDataBatchType.Validation: self.generate_validation_data}
        generator = mapping[batch_data_type]()
        return generator


class ModelDataBatchType(Enum):
    Training = 1
    Validation = 2
    Test = 3

# how to use dictionary generator
# data = ClickStreamGenerator(1, 10, 2)
# data_generator = data.get_generator(ModelDataBatchType.Training)
# print(next(data_generator))

# data = ClickStreamGenerator(1, 10, 2)
# print(next(data.generate_train_data()))