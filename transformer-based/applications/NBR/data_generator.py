import numpy as np
import os
from applications.NBR.NBR_constant import SEQ_LEN, MIN_SEQ_LEN
from sequence_transformer.constants import INPUT_PADDING_TOKEN


class ClickStreamGenerator:
    def __init__(self, n_items, write_vocab_files=False, vocab_dir=None):
        """

        Args:
            n_items:
            n_events:
            session_cohesiveness: The cosine similarities of items is scaled by this factor before being softmax-ed.
                Higher values will result in transitions to closer items, i.e. the probability of transitioning to items
                that are further away diminish more strongly for higher values of this.
        """
        self._min_basket_len = 1
        self._max_basket_len = 5
        self._n_items = n_items

        self.item_names = [f'item_{i}' for i in range(self._n_items)]

        if write_vocab_files:
            item_vocab_filename = os.path.join(vocab_dir, 'item_vocab.txt')
            with open(item_vocab_filename, 'w') as f:
                f.writelines('\n'.join(self.item_names))

    def __iter__(self):
        return self

    # tf.data's from_generator requires a callable generator. This makes the object callable
    def __call__(self, *args, **kwargs):
        return self.__next__()

    def _multi_hop_converter(self, items, size):
        one_hoted = np.zeros(size)
        for item in items:
            one_hoted[int(item.split('_')[-1])] = 1.0
        return list(one_hoted)  # list( map(float, one_hoted))

    def __next__(self):

        userID = f'user_{np.random.choice(range(1000))}'
        basket_size = np.random.randint(self._min_basket_len, self._max_basket_len + 1)
        # basket = [f'item_{item}' for item in np.random.choice(range(self._n_items), basket_size, replace=False)]
        basket = [self.item_names[item] for item in np.random.choice(range(self._n_items), basket_size, replace=False)]

        # TODO: right now all the sample has SEQ_LEN baskets. Figure out the variable session length
        # There is a error in input pipeline when I do so
        session_length = np.random.randint(MIN_SEQ_LEN, SEQ_LEN+1)
        # EROOR: TypeError: `generator` yielded an element that did not match the expected structure. The expected structure was {'userID': tf.string, 'feature1': tf.string, 'feature2': tf.string, 'feature3': tf.string, 'feature4': tf.string, 'feature5': tf.string, 'feature6': tf.string, 'feature7': tf.string, 'feature8': tf.string, 'label': tf.float32}, but the yielded element was {'userID': '923', 'feature1': ['4', '5', '2'], 'feature2': ['4', '5', '2'], 'feature3': ['4', '5', '2'], 'feature4': ['4', '5', '2'], 'feature5': ['4', '5', '2'], 'feature6': ['4', '5', '2'], 'label': [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]}.

        data ={'userID': userID}
        # for i in range(session_length):
        for i in range(SEQ_LEN-1):
            data[f'feature{i+1}'] = basket
            # label
        # for i in range(session_length, SEQ_LEN):
        #     data[f'feature{i+1}'] = INPUT_PADDING_TOKEN
        #     # label

        data['label'] = self._multi_hop_converter(basket, self._n_items)

        yield data


if __name__ == '__main__':
    N_ITEMS = 10
    COHESION = 1
    item_vocab_dir = 'data/simulated/vocabs'
    if not os.path.exists(item_vocab_dir):
        os.makedirs(item_vocab_dir)

    data_src = ClickStreamGenerator(n_items=N_ITEMS, write_vocab_files=True, vocab_dir=item_vocab_dir)
    for data in next(data_src):
        print(data)
