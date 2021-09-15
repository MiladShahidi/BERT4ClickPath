import tensorflow as tf
from sequence_transformer.constants import LABEL_PAD


class PositiveRate(tf.keras.metrics.Metric):

    def __init__(self, name='positive_rate', **kwargs):
        super(PositiveRate, self).__init__(name=name, **kwargs)
        self.n_returned_items = self.add_weight(name='positive_rate', initializer='zeros')
        self.n_items = self.add_weight(name='n_items', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
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

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)  # threshold = 0.5

        tp = tf.logical_and(tf.cast(y_true, tf.int32) == 1, tf.cast(y_pred, tf.int32) == 1)
        tp = tf.cast(tp, dtype=tf.float32)

        condition_true = (tf.cast(y_true, tf.int32) == 1)
        condition_true = tf.cast(condition_true, dtype=tf.float32)

        predicted_true = (tf.cast(y_pred, tf.int32) == 1)
        predicted_true = tf.cast(predicted_true, dtype=tf.float32)
        self.tp.assign_add(tf.reduce_sum(tp))
        self.condition_true.assign_add(tf.reduce_sum(condition_true))
        self.predicted_true.assign_add(tf.reduce_sum(predicted_true))

    def result(self):
        return 2 * self.tp / (self.condition_true+self.predicted_true)

    def reset_states(self):
        self.tp.assign(0.)
        self.condition_true.assign(0.)
        self.predicted_true.assign(0.)


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

