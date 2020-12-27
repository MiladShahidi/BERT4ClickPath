import numpy as np
import os
from constants import INPUT_MASKING_TOKEN


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def normalize_matrix(a, axis=1):
    axis_norms = np.linalg.norm(a, ord=2, axis=axis, keepdims=True)
    return np.divide(a, axis_norms)


class ReturnsDataGen:

    def __init__(self, n_items, n_events, session_cohesiveness, positive_rate, write_vocab_files=False, vocab_dir=None):
        """

        Args:
            n_items:
            n_events:
            session_cohesiveness: The cosine similarities of items is scaled by this factor before being softmax-ed.
                Higher values will result in transitions to closer items, i.e. the probability of transitioning to items
                that are further away diminish more strongly for higher values of this.
        """
        self._min_sess_len = 5
        self._max_sess_len = 30
        self._min_basket_len = 1
        self._max_basket_len = 5
        self._n_items = n_items
        self._n_events = n_events
        self.positive_rate = positive_rate
        # TODO: Write vocab_files or create IO streams

        self.item_names = [f'item_{i}' for i in range(self._n_items)]
        self.event_names = [f'event_{i}' for i in range(self._n_events)]

        if write_vocab_files:
            item_vocab_filename = os.path.join(vocab_dir, 'item_vocab.txt')
            with open(item_vocab_filename, 'w') as f:
                f.writelines('\n'.join(self.item_names))
            event_vocab_filename = os.path.join(vocab_dir, 'event_vocab.txt')
            with open(event_vocab_filename, 'w') as f:
                f.writelines('\n'.join(self.event_names))

        # Creating the item embedding space and the transition matrix
        true_item_repr_dims = 5
        # 1) ensure that each dimension is centered around 0 (origin) by setting low=-1, high=1, i.e. points in a cube
        true_item_repr_matrix = np.random.uniform(low=-1, high=1, size=(n_items, true_item_repr_dims))
        # 2) Map the points onto the surface of the unit sphere (around origin) by normalizing their L2 norm to 1
        normalized_item_repr_matrix = normalize_matrix(true_item_repr_matrix, axis=1)
        # Now the dot product will be equal to cosine of the angle between the vectors because they all have norm=0
        cosine_similarity = np.matmul(normalized_item_repr_matrix,
                                      np.transpose(normalized_item_repr_matrix))
        # The diagonal elements of cosine similarity matrix will always be 1. subtracting 2 * I will turn these to -1
        # This avoids transitioning from an item to itself. Otherwise, we'll get stuck on one item for most of the time
        self_masked_cosine_sim = cosine_similarity - 2 * np.eye(cosine_similarity.shape[0])
        self.transition_matrix = softmax(session_cohesiveness * self_masked_cosine_sim, axis=1)

    def __iter__(self):
        return self

    # tf.data's from_generator requires a callable generator. This makes the object callable
    def __call__(self, *args, **kwargs):
        return self.__next__()

    def _draw_sample_sequence(self, starting_point, n_samples):
        current_item = starting_point
        sample_seq = [current_item]
        for _ in range(n_samples):
            transition_prob_dist = self.transition_matrix[current_item, :]
            next_item = np.random.choice(range(self._n_items), p=transition_prob_dist)
            sample_seq.append(next_item)
            current_item = next_item

        return np.array(sample_seq)

    def _mask_session(self, session):
        session_items = session.copy()
        num_masked = np.random.randint(1, self._max_basket_len)
        mask_index = np.random.choice(range(session_items.shape[0]), size=num_masked, replace=False)

        positive_examples = session_items[mask_index]
        negative_examples = np.random.choice(range(self._n_items), size=num_masked)

        pos_probs = np.random.uniform(size=num_masked)

        basket = np.where(pos_probs < self.positive_rate, positive_examples, negative_examples)
        labels = np.where(pos_probs < self.positive_rate, 1.0, 0.0)

        session_items[mask_index] = -1

        return session_items, basket, labels

    def __next__(self):
        session_length = np.random.randint(self._min_sess_len, self._max_sess_len)

        session_items = self._draw_sample_sequence(np.random.choice(range(self._n_items)), n_samples=session_length)
        masked_session, basket, labels = self._mask_session(session_items)

        side_feature = np.random.random()

        # n_baskets = np.random.randint(low=2, high=10, dtype=np.int32)
        # basket_sizes = np.random.randint(low=2, high=10, size=n_baskets, dtype=np.int32)
        # list_of_lists = [list(np.random.uniform(size=basket_size)) for basket_size in basket_sizes]

        data = {
            'seq_1_items': [self.item_names[i] if i >= 0 else INPUT_MASKING_TOKEN for i in masked_session],
            'seq_1_events': [self.event_names[0] for i in masked_session],  # Just a placeholder for now
            'seq_2_items': [self.item_names[i] for i in basket],
            'seq_2_events': [self.event_names[0] for i in basket],  # Just a silly place holder for now
            'side_feature_1': side_feature,
            # 'list_of_lists': list_of_lists,
            'label': labels
        }

        yield data


if __name__ == '__main__':
    data_gen = ReturnsDataGen(4, 4, 5, 0.3)
