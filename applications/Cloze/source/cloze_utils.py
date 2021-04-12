import tensorflow as tf
from sequence_transformer.constants import LABEL_PAD
from sequence_transformer.losses import MaskedLoss


def cloze_output_adaptor(y_true, y_pred):
    """
    This helper function reshapes and removes pads from y_true and y_pred before sending them to loss and metrics.

    This is only needed in distributed training. The masked versions of losses and metrics will work fine on their own
    when training on a single GPU/CPU.

    In the Cloze task (masked language model) we mask a (variable) number of items in each sequence and the model
    tries to predict what that item was. So,
    y_true is (batch_size, max_n_masked)  # true item numbers as integers
    y_pred is (batch_size, max_n_masked, vocab_size)  # The output is prob. dist over vocabulary

    where max_n_masked is the maximum number of masked positions in the batch. All examples (and y_true) are padded to
    the longest in the batch. So if one of the examples has 5 masked positions and all others have fewer, y_true will
    be padded to (batch_size, 5). A typical row of y_true may look like: [1, 3, -1, -1, -1] where -1 is the pad.

    For the purpose of calculating loss and metrics, each masked position is independent from others and it doesn't
    matter which sequence they belong to. So we can merge the batch dimension with the second dimension and then
    remove the pads from both y_true and y_pred.

    For training on a single GPU or CPU, this is not necessary. The reason we would even bother to do this reshaping
    has to do with what happens in distributed training. In distributed training, y_true is padded before splitting the
    batch across GPUs, but y_pred gets padded inside the model. That is, each GPU will pad it to the longest example
    in **its own (sub)batch** (see the call method of the SequenceTransformer class). The result is that y_true may get
    padded to 5, because the maximum number of masked positions in the GLOBAL batch was 5, but one GPU may only pad to 3
    because the maximum number of masked positions it can see in its own (sub)batch is 3. And these two shapes are
    incompatible for loss and metric calculations.

    The solution is to:
    1) reshape them to
        y_ture (n_examples, 1)
        y_pred (n_examples, vocab_size)
        where n_examples = batch_size * max_n_masked.

    2) use y_true to create a padding mask and remove all pads from both. Then they will have compatible shapes.

    The result will be sent to loss and metric functions.

    Args:
        y_true:
        y_pred:

    Returns:

    """
    # Merge the first two dimensions. Each example may contain more than 1 masked item
    # But we treat all of them the same way. So we will flatten the batch dimension together with the mask dim.
    dim_per_example = tf.shape(y_pred)[-1]  # e.g. vocab_size when the model outputs a distribution over vocabulary
    y_pred = tf.reshape(y_pred, (-1, dim_per_example))  # (n_examples, k)  n_examples = batch_size * max_n_masked
    y_true = tf.reshape(y_true, (-1, 1))  # (n_examples, 1)  n_examples = batch_size * max_n_masked

    # This again assumes only 1 item in true_ranking. the result will be a 1-D tensor
    boolean_mask = tf.squeeze(tf.where(y_true == tf.cast(LABEL_PAD, y_true.dtype), False, True), axis=1)
    y_true = tf.boolean_mask(y_true, boolean_mask)
    y_pred = tf.boolean_mask(y_pred, boolean_mask)

    return y_true, y_pred


class ClozeMaskedLoss(tf.keras.losses.Loss):
    def __init__(self, item_wise_loss_fn, label_pad=LABEL_PAD):
        """
        see cloze_output_adaptor for how this works and why it is needed.

        Args:
            item_wise_loss_fn: An item-wise loss function that does not perform any reduction (e.g. mean over batch).
                For example, `tf.keras.backend.binary_crossentropy` is one such loss function.
                NOTE: usual loss functions like `tf.keras.losses.binary_crossentropy` performs reduction
            pos_weight: weight for positive class for binary labels. Only works as expected for binary labels.
        """
        super(ClozeMaskedLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE)
        self.masked_loss = MaskedLoss(item_wise_loss_fn=item_wise_loss_fn, label_pad=label_pad)

    def call(self, y_true, y_pred):
        y_true, y_pred = cloze_output_adaptor(y_true, y_pred)
        # Even though pads have been removed in the previous step, but the masked loss class below is safer to use
        # because in addition to handling masks, it is also distribution-aware and will avoid nan (div-by-0) loss
        return self.masked_loss(y_true, y_pred)


class ClozeMaskedRecall(tf.keras.metrics.Metric):
    """
    WARNING:
    This was written with Cloze (masked language model) task in mind. It might not be generalizable to other settings.
    In particular this assumes that the true ranking is a list of only 1 item. Because in the recommendation or
    masked sequence task, each example has only 1 ground truth item.
    """

    def __init__(self, k, name=None):
        if name is None:
            name = f'Recall_at_{k}'
        super(ClozeMaskedRecall, self).__init__(name=name)
        self.k = k
        self.n_examples = self.add_weight(name='n_examples', initializer='zeros')
        self.recall = self.add_weight(name='n_examples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # This was written with Cloze (masked language model) or recommendation from click-stream in mind
        # It assumes that the ground truth (true ranking) for each masked position consists of only 1 item
        # y_true: (batch_size, max_n_masked) - item numbers as integers
        # y_pred: (batch_size, max_n_masked, vocab_size) - predicted probabilities over the vocabulary
        # max_n_masked = Maximum no. masked items in an example in the batch. Other examples are padded to this length

        y_true, y_pred = cloze_output_adaptor(y_true, y_pred)

        # After cloze_output_adaptor:
        # y_ture: (n_examples, 1)
        # y_pred: (n_examples, vocab_size)
        # where n_examples = batch_size * max_n_masked.

        # Indices (i.e. item no) of top k scores along the last axis
        _, pred_ranking = tf.math.top_k(input=y_pred, k=self.k)  # (n_examples, k) see the comment above
        pred_ranking = tf.cast(pred_ranking, y_true.dtype)  # They need to have the same dtype for the comparison below

        n_actual_examples = tf.shape(y_true)[0]  # We record number of actual examples after removing pads

        # tf.equal operates along the first dimension, even though they have different sizes on the 2nd axis
        relevant = tf.where(tf.equal(pred_ranking, y_true), 1.0, 0.0)  # relevant=0 or 1. whether the item = true_item?
        # relevant is (n_exmaples, k)
        recall = tf.reduce_sum(relevant, axis=1)  # 0 or 1 for each masked position: was the true item among top k?

        self.recall.assign_add(tf.reduce_sum(recall))  # reduce over batch
        self.n_examples.assign_add(tf.cast(n_actual_examples, tf.float32))  # needs to be float for division in result

    def result(self):
        return self.recall / self.n_examples

    def reset_states(self):
        self.n_examples.assign(0)
        self.recall.assign(0.)


class ClozeMaskedNDCG(tf.keras.metrics.Metric):
    """
    This was written with Cloze (masked language model) in mind. It might not be generalizable to other settings.
    In particular this assumes that the true ranking is a list of only 1 item. Because in the recommendation or
    masked sequence task, each example has only 1 ground truth item.
    """

    def __init__(self, k, name=None):
        if name is None:
            name = f'NDCG_at_{k}'
        super(ClozeMaskedNDCG, self).__init__(name=name)
        self.k = k
        self.n_examples = self.add_weight(name='n_examples', initializer='zeros')
        self.ndcg = self.add_weight(name='n_examples', initializer='zeros')
        self.discount_weights = 1 / self.log_base_2(tf.range(2, k + 2, dtype=tf.float32))

    @staticmethod
    def log_base_2(x):
        return tf.math.log(x) / tf.math.log(2.0)

    def _dcg(self, ranking, true_ranking):
        # This is written only for the case where true_ranking contains only 1 item for each example
        gains = tf.where(tf.equal(ranking, true_ranking), 1.0, 0.0)  # gain=0 or 1. whether the item = true_item?
        # when true ranking is passed to this function, it may contain < k items
        n_items = tf.shape(ranking)[1]  # ranking is (batch_size, n_items)
        weights = self.discount_weights[:n_items]

        res = tf.reduce_sum(gains * weights, axis=1)
        return res

    def update_state(self, y_true, y_pred, sample_weight=None):
        # This was written with Cloze (masked language model) or recommendation from click-stream in mind
        # It assumes that the ground truth (true ranking) for each masked position consists of only 1 item
        # y_true: (batch_size, max_n_masked) - item numbers as integers
        # y_pred: (batch_size, max_n_masked, vocab_size) - predicted probabilities over the vocabulary
        # max_n_masked = Maximum no. masked items in an example in the batch. Other examples are padded to this length

        y_true, y_pred = cloze_output_adaptor(y_true, y_pred)

        # Indices (i.e. item no) of top k scores along the last axis
        _, pred_ranking = tf.math.top_k(input=y_pred, k=self.k)  # (batch_size, max_n_masked, k)
        pred_ranking = tf.cast(pred_ranking, y_true.dtype)  # They need to have the same dtype in dcg calculations

        n_actual_examples = tf.shape(y_true)[0]  # We record number of actual examples after removing pads

        ndcg = self._dcg(pred_ranking, y_true) / self._dcg(y_true, y_true)  # (n_examples,)
        self.ndcg.assign_add(tf.reduce_sum(ndcg))  # reduce over batch
        self.n_examples.assign_add(tf.cast(n_actual_examples, tf.float32))  # needs to be float for division in result

    def result(self):
        return self.ndcg / self.n_examples

    def reset_states(self):
        self.n_examples.assign(0)
        self.ndcg.assign(0.)


if __name__ == '__main__':
    y_true = [[1, 0]]
    y_pred = [[
        [0.9, 0.1, 0.01],
        [0.5, 0.3, 0.01]
    ]]
    # metric
    ndcg = ClozeMaskedNDCG(k=3)
    ndcg.update_state(y_true, y_pred)
    a = ndcg.result()
    print('my metric:', a)
    y_true = [[0, 1, 0], [1, 0, 0]]
    y_pred = [[0.9, 0.1, 0.01], [0.5, 0.3, 0.01]]
    from sklearn.metrics import ndcg_score
    sk_res = ndcg_score(y_true, y_pred, k=3)
    print(sk_res)
