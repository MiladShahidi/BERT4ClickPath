import numpy as np


class ReturnsDataGen:

    def __init__(self, n_items, n_events):

        self._min_sess_len = 5
        self._max_sess_len = 30
        self._min_basket_len = 5
        self._max_basket_len = 30
        self._n_items = n_items
        self._n_events = n_events
        # TODO: Write vocab_files or create IO streams

    def __iter__(self):
        return self

    # TF's from_generator requires a callable generator. This makes the object callable
    def __call__(self, *args, **kwargs):
        return self.__next__()

    def __next__(self):
        item_names = [f'item_{i}' for i in range(self._n_items)]
        event_names = [f'item_{i}' for i in range(self._n_events)]

        session_length = np.random.randint(self._min_sess_len, self._max_sess_len)
        basket_length = np.random.randint(self._min_basket_len, self._max_basket_len)

        session_items = np.random.choice(item_names, size=session_length)
        session_events = np.random.choice(event_names, size=session_length)

        basket_items = np.random.choice(item_names, size=basket_length)
        basket_events = np.random.choice(event_names, size=basket_length)

        side_feature = np.random.random()

        label = np.round(np.random.random(size=basket_length))  # 0 or 1

        data = {
            'seq_1_items': session_items,
            'seq_1_events': session_events,
            'seq_2_items': basket_items,
            'seq_2_events': basket_events,
            'side_feature_1': side_feature,
            'label': label
        }

        yield data
