import tensorflow as tf
from sequence_transformer.constants import LABEL_PAD


class PositiveRate(tf.keras.metrics.Metric):

    def __init__(self, name='positive_rate', **kwargs):
        super(PositiveRate, self).__init__(name=name, **kwargs)
        self.n_returned_items = self.add_weight(name='positive_rate', initializer='zeros')
        self.n_items = self.add_weight(name='n_items', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_pred = tf.round(y_pred)  # threshold = 0.5
        mask = tf.math.logical_not(tf.math.equal(y_true, LABEL_PAD))
        mask = tf.cast(mask, dtype=y_true.dtype)

        masked_y_true = y_true * mask

        self.n_returned_items.assign_add(tf.reduce_sum(masked_y_true))
        self.n_items.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.n_returned_items / self.n_items

    def reset_states(self):
        self.n_returned_items.assign(0.)
        self.n_items.assign(0.)


class PredictedPositives(tf.keras.metrics.Metric):

    def __init__(self, name='pred_positives', **kwargs):
        super(PredictedPositives, self).__init__(name=name, **kwargs)
        self.n_pred_returns = self.add_weight(name='pred_returned', initializer='zeros')
        self.n_items = self.add_weight(name='n_items', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)  # threshold = 0.5
        mask = tf.math.logical_not(tf.math.equal(y_true, LABEL_PAD))
        mask = tf.cast(mask, dtype=y_true.dtype)

        masked_y_pred = y_pred * mask

        self.n_pred_returns.assign_add(tf.reduce_sum(masked_y_pred))
        self.n_items.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.n_pred_returns / self.n_items

    def reset_states(self):
        self.n_pred_returns.assign(0.)
        self.n_items.assign(0.)


class F1Score(tf.keras.metrics.Metric):
    # ToDo: TF doesn't have an F1 metric (tfa does, but didn't want to use that).
    #  This used to be MaskedF1. After we created the wrapper MaskerMetric class I changed this to be a normal F1,
    #  so that it can be wrapped by that class but didn't test it.
    #  Test this before using it.
    def __init__(self, name='F1Score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.condition_true = self.add_weight(name='condition_true', initializer='zeros')
        self.predicted_true = self.add_weight(name='pred_true', initializer='zeros')
        # self.n_items = self.add_weight(name='n_items', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)  # threshold = 0.5

        tp = tf.logical_and(tf.cast(y_true, tf.int32) == 1, tf.cast(y_pred, tf.int32) == 1)
        tp = tf.cast(tp, dtype=tf.float32)
        # mask = tf.math.logical_not(tf.math.equal(y_true, LABEL_PAD))
        # mask = tf.cast(mask, dtype=tp.dtype)
        # tp *= mask
        # tp = tf.cast(tp, tf.float32)

        condition_true = (tf.cast(y_true, tf.int32) == 1)
        condition_true = tf.cast(condition_true, dtype=tf.float32)
        # condition_true *= mask
        # condition_true = tf.cast(condition_true, tf.float32)

        predicted_true = (tf.cast(y_pred, tf.int32) == 1)
        predicted_true = tf.cast(predicted_true, dtype=tf.float32)
        # predicted_true *= mask
        # predicted_true = tf.cast(predicted_true, tf.float32)
        self.tp.assign_add(tf.reduce_sum(tp))
        self.condition_true.assign_add(tf.reduce_sum(condition_true))
        self.predicted_true.assign_add(tf.reduce_sum(predicted_true))

    def result(self):
        return 2 * self.tp / (self.condition_true+self.predicted_true)

    def reset_states(self):
        self.tp.assign(0.)
        self.condition_true.assign(0.)
        self.predicted_true.assign(0.)


class fbeta_2(tf.keras.metrics.Metric):
    def __init__(self, name='fbeta', **kwargs):
        super(fbeta_2, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
        self.beta = 1
        self.precision = self.add_weight(name='precision', initializer='zeros')
        self.recall = self.add_weight(name='recall', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # print('*'*10, self.beta)
        y_pred = tf.keras.backend.clip(y_pred, 0, 1)
        # calculate elements
        tp = tf.reduce_sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        fp = tf.reduce_sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred - y_true, 0, 1)))
        fn = tf.reduce_sum(tf.keras.backend.round(tf.keras.backend.clip(y_true - y_pred, 0, 1)))
        # calculate precision

        self.precision.assign_add(tf.reduce_sum(tp / (tp + fp + tf.keras.backend.epsilon())))
        # calculate recall
        self.recall.assign_add(tf.reduce_sum(tp / (tp + fn + tf.keras.backend.epsilon())))
        # calculate fbeta, averaged across each class

    def result(self):
        # return tf.keras.backend.mean((1 + self.beta**2) * (self.precision * self.recall) / (self.beta**2 * self.precision + self.recall + tf.keras.backend.epsilon()))
        return (1 + self.beta**2) * (self.precision * self.recall) / (self.beta**2 * self.precision + self.recall + tf.keras.backend.epsilon())

    def reset_states(self):
        self.tp.assign(0.)
        self.fp.assign(0.)
        self.fn.assign(0.)
        self.precision.assign(0.)
        self.recall.assign(0.)
        # self.beta = 1


class MaskedMetric(tf.keras.metrics.Metric):

    def __init__(self, metric, name, **kwargs):
        super(MaskedMetric, self).__init__(name=name, **kwargs)
        self._metric = metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("Masked metrics do not support sample_weight.")

        mask = tf.logical_not(tf.equal(y_true, tf.cast(LABEL_PAD, y_true.dtype)))
        self._metric.update_state(y_true, y_pred, sample_weight=mask)

    def result(self):
        return self._metric.result()

    def reset_states(self):
        self._metric.reset_states()


def fbeta(y_true, y_pred, beta=1):
    """
    Calculate f-beta score for multi-class/label classification. Implementation from https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/
    Args:
        y_true:
        y_pred:
        beta:

    Returns:

    """
    # clip predictions
    y_pred = tf.keras.backend.clip(y_pred, 0, 1)
    # calculate elements
    tp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true - y_pred, 0, 1)), axis=1)
    # calculate precision
    p = tp / (tp + fp + tf.keras.backend.epsilon())
    # calculate recall
    r = tp / (tp + fn + tf.keras.backend.epsilon())
    # calculate fbeta, averaged across each class
    bb = beta ** 2
    fbeta_score = tf.keras.backend.mean((1 + bb) * (p * r) / (bb * p + r + tf.keras.backend.epsilon()))
    return fbeta_score


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
