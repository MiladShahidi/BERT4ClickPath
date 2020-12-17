import tensorflow as tf
from constants import LABEL_PAD
import math
# from focal_loss import SigmoidFocalCrossEntropy


# def custom_lr_schedule_fn(d_model, warmup_steps=4000, scale=1):
#     def _lr_schedule(step):
#         arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
#         arg2 = step * (warmup_steps ** -1.5)
#         learning_rate = tf.math.rsqrt(tf.cast(d_model, tf.float32)) * tf.math.minimum(arg1, arg2) * scale
#
#         return learning_rate * scale
#
#     return _lr_schedule


class CustomLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, scale=1):
        super(CustomLRSchedule, self).__init__()

        self.d_model = float(d_model)  # Don't tf.cast this. It will result in: "Tensor object not json serializable"
        self.warmup_steps = warmup_steps
        self.scale = scale

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
            'scale': self.scale
        }
        return config

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        learning_rate = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) * self.scale

        return learning_rate * self.scale


class MaskedBinaryAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='accuracy', **kwargs):
        super(MaskedBinaryAccuracy, self).__init__(name=name, **kwargs)
        self.correctly_classified = self.add_weight(name='n_correct', initializer='zeros')
        self.n_items = self.add_weight(name='n_items', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)  # threshold = 0.5
        matches = (tf.cast(y_true, tf.int32) == tf.cast(y_pred, tf.int32))
        matches = tf.cast(matches, dtype=tf.float32)

        mask = tf.math.logical_not(tf.math.equal(y_true, LABEL_PAD))
        mask = tf.cast(mask, dtype=matches.dtype)

        matches *= mask
        matches = tf.cast(matches, tf.float32)

        self.correctly_classified.assign_add(tf.reduce_sum(matches))
        self.n_items.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.correctly_classified / self.n_items

    def reset_states(self):
        self.correctly_classified.assign(0.)
        self.n_items.assign(0.)


class MaskedPrecision(tf.keras.metrics.Metric):

    def __init__(self, name='masked_precision', **kwargs):
        super(MaskedPrecision, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.predicted_true = self.add_weight(name='pred_true', initializer='zeros')
        # self.n_items = self.add_weight(name='n_items', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)  # threshold = 0.5
        tp = tf.logical_and(tf.cast(y_true, tf.int32) == 1, tf.cast(y_pred, tf.int32) == 1)
        tp = tf.cast(tp, dtype=tf.float32)
        mask = tf.math.logical_not(tf.math.equal(y_true, LABEL_PAD))
        mask = tf.cast(mask, dtype=tp.dtype)
        tp *= mask
        tp = tf.cast(tp, tf.float32)

        predicted_true = (tf.cast(y_pred, tf.int32) == 1)
        predicted_true = tf.cast(predicted_true, dtype=tf.float32)
        predicted_true *= mask
        predicted_true = tf.cast(predicted_true, tf.float32)

        self.tp.assign_add(tf.reduce_sum(tp))
        self.predicted_true.assign_add(tf.reduce_sum(predicted_true))

    def result(self):
        return self.tp / self.predicted_true

    def reset_states(self):
        self.tp.assign(0.)
        self.predicted_true.assign(0.)


class MaskedRecall(tf.keras.metrics.Metric):

    def __init__(self, name='masked_recall', **kwargs):
        super(MaskedRecall, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.condition_true = self.add_weight(name='condition_true', initializer='zeros')
        # self.n_items = self.add_weight(name='n_items', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)  # threshold = 0.5
        tp = tf.logical_and(tf.cast(y_true, tf.int32) == 1, tf.cast(y_pred, tf.int32) == 1)
        tp = tf.cast(tp, dtype=tf.float32)
        mask = tf.math.logical_not(tf.math.equal(y_true, LABEL_PAD))
        mask = tf.cast(mask, dtype=tp.dtype)
        tp *= mask
        tp = tf.cast(tp, tf.float32)

        condition_true = (tf.cast(y_true, tf.int32) == 1)
        condition_true = tf.cast(condition_true, dtype=tf.float32)
        condition_true *= mask
        condition_true = tf.cast(condition_true, tf.float32)

        self.tp.assign_add(tf.reduce_sum(tp))
        self.condition_true.assign_add(tf.reduce_sum(condition_true))

    def result(self):
        return self.tp / self.condition_true

    def reset_states(self):
        self.tp.assign(0.)
        self.condition_true.assign(0.)


class MaskedF1(tf.keras.metrics.Metric):

    def __init__(self, name='F1', **kwargs):
        super(MaskedF1, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.condition_true = self.add_weight(name='condition_true', initializer='zeros')
        self.predicted_true = self.add_weight(name='pred_true', initializer='zeros')
        # self.n_items = self.add_weight(name='n_items', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)  # threshold = 0.5
        tp = tf.logical_and(tf.cast(y_true, tf.int32) == 1, tf.cast(y_pred, tf.int32) == 1)
        tp = tf.cast(tp, dtype=tf.float32)
        mask = tf.math.logical_not(tf.math.equal(y_true, LABEL_PAD))
        mask = tf.cast(mask, dtype=tp.dtype)
        tp *= mask
        tp = tf.cast(tp, tf.float32)

        condition_true = (tf.cast(y_true, tf.int32) == 1)
        condition_true = tf.cast(condition_true, dtype=tf.float32)
        condition_true *= mask
        condition_true = tf.cast(condition_true, tf.float32)

        predicted_true = (tf.cast(y_pred, tf.int32) == 1)
        predicted_true = tf.cast(predicted_true, dtype=tf.float32)
        predicted_true *= mask
        predicted_true = tf.cast(predicted_true, tf.float32)

        self.tp.assign_add(tf.reduce_sum(tp))
        self.condition_true.assign_add(tf.reduce_sum(condition_true))
        self.predicted_true.assign_add(tf.reduce_sum(predicted_true))

    def result(self):
        return 2 * self.tp / (self.condition_true+self.predicted_true)

    def reset_states(self):
        self.tp.assign(0.)
        self.condition_true.assign(0.)
        self.predicted_true.assign(0.)


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


class MaskedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, pos_weight=None):
        super(MaskedBinaryCrossEntropy, self).__init__(reduction=tf.keras.losses.Reduction.NONE)
        self.pos_weight = tf.cast(pos_weight, tf.float32) if pos_weight is not None else None
        self._negative_weight = 1.0  # This is only to make the code more readable and possibly easier to extend later

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: (batch_size, input_seq_len). Tokens that do not have a label are represented by the constant
                LABEL_PAD (=-1) and should be ignored when calculating the loss.
            y_pred:

        Returns:
        """
        # loss_calculator = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        # Note: I'm using the BCE function (below) instead of the object above, because that object returns the mean loss
        # per example, while I need individual losses for output nodes because I need to mask the padded ones before
        # averaging them. (additional note: the reduction argument above prevents averaging across batches, not per node)

        # Note this is backend and not tf.keras.losses.binary_crossentropy. The difference is that this one does not
        # perform any reduction while the one in the losses module, even when reduction=None, reduces the last dimension

        item_loss = tf.keras.backend.binary_crossentropy(y_true, y_pred,
                                                         from_logits=False)  # (batch_size, input_seq_len)

        # # # Mask padded items
        mask = tf.math.logical_not(tf.math.equal(y_true, LABEL_PAD))
        mask = tf.cast(mask, dtype=item_loss.dtype)
        item_loss *= mask  # padded entries are multiplied by 0 (masked)

        if self.pos_weight is not None:
            sample_weight = tf.where(tf.equal(y_true, 1), self.pos_weight, self._negative_weight)
            item_loss = tf.multiply(sample_weight, item_loss)

        # # # Reduction
        # Option 1: Give all items equal weight. i.e. the loss from the only item of a 1-item seq_2 is given the same
        # weight as that of each item in a 2-item seq_2. So we just average over all items in the batch
        total_loss = tf.reduce_sum(item_loss)
        n_items = tf.reduce_sum(mask)

        # In a multi-GPU setup, each global batch will be split between GPUs and if batch size varies, or is not a
        # multiple of number of GPUs, some GPUs will receive empty batches. This can cause nan loss. One work-around
        # is to explicitly handle this case here.
        # https://stackoverflow.com/questions/54283937/training-on-multiple-gpus-causes-nan-validation-errors-in-keras
        # This does not happen during training. My guess is that because training data is repeated indefinitely, there
        # is never a last smaller batch with a different size and all batches are the same size (which we set to a
        # multiple of number of GPUs manually).
        mean_batch_loss = tf.cond(pred=tf.size(y_true) == 0,
                                  true_fn=lambda: 0.0,  # Avoid dividing by 0 if batch is empty.
                                  false_fn=lambda: tf.divide(total_loss, n_items))

        # # # Normalize for weights. Optional. But it will keep the loss on the same scale as unweighted loss
        if self.pos_weight is not None:
            weight_normalization = (self.pos_weight + self._negative_weight) / 2
            mean_batch_loss = mean_batch_loss / weight_normalization

        # Option 2: First average the loss within each seq_2, then average over batch. This will give higher weight to
        # items in smaller baskets, because the loss of each item in a large seq_2 will be divided by a larger number
        # This is not so appealing since 1) why give more weight to smaller baskets? 2) The relative weight given to baskets
        # with different sizes will depend on the sizes of baskets that happen to be in that batch

        # Note: I changed this one at some point, so it returns a (batch_size, ...) tensor, instead of a scalar.
        # total_per_ex_loss = tf.reduce_sum(item_loss, axis=1, keepdims=True, name='total_per_ex_loss')  # (batch_size, 1)
        # basket_sizes = tf.reduce_sum(mask, axis=1, keepdims=True, name='basket_sizes')  # (batch_size, 1)
        # masked_loss = tf.divide(total_per_ex_loss, basket_sizes, name='masked_loss')
        # mean_batch_loss = tf.reduce_mean(ave_per_ex_loss)

        return mean_batch_loss


class MaskedFocalLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        from_logits: bool = False,
        alpha=0.25,
        gamma=2.0,
        reduction: str = tf.keras.losses.Reduction.NONE,
        name: str = "sigmoid_focal_crossentropy",
    ):
        """
        Args:
            class_weights: A dictionary mapping class labels to class weights
        """
        # It's important to set reduction to None because we want to take mask into account before reducing the loss
        self.from_logits = from_logits
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.name = name

        super(MaskedFocalLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE)
        self.focal_loss_fn = SigmoidFocalCrossEntropy(
            from_logits=from_logits,
            alpha=alpha,
            gamma=gamma,
            reduction=reduction,
            name=name
        )

    def get_config(self):
        config = super(MaskedFocalLoss, self).get_config()
        config.update({
            'from_logits': self.from_logits,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'reduction': self.reduction,
            'name': self.name
        })
        return config

    def call(self, y_true, y_pred, from_logits=False):
        """
        Args:
            y_true: (batch_size, input_seq_len). Tokens that do not have a label are represented by the constant
                LABEL_PAD (=-1) and should be ignored when calculating the loss.
                These include [PAD] entries as well as items from click-stream
            y_pred:
            from_logits:

        Returns:
        """
        # Note: I'm using a loss function instead of classes (e.g.) tf.keras.losses.BinaryCrossentropy,
        # because for some losses, the object doesn't let you control the reduction method and returns the mean loss per
        # example, while I need individual losses for output nodes because I need to mask the padded ones before
        # averaging them. (note: the reduction argument above prevents averaging across batches, not per node)

        mask = tf.math.logical_not(tf.math.equal(y_true, LABEL_PAD))
        masked_item_loss = self.focal_loss_fn(y_true=y_true, y_pred=y_pred, sample_weight=mask)  # (batch_size, input_seq_len)
        total_loss = tf.reduce_sum(masked_item_loss)

        mask = tf.cast(mask, tf.float32)
        n_items = tf.reduce_sum(mask)

        # In a multi-GPU setup, each global batch will be split between GPUs and if batch size varies, or is not a
        # multiple of number of GPUs, some GPUs will receive empty batches. This can cause nan loss. One work-around
        # is to explicitly handle this case here.
        # https://stackoverflow.com/questions/54283937/training-on-multiple-gpus-causes-nan-validation-errors-in-keras
        # This does not happen during training. My guess is that because training data is repeated indefinitely, there
        # is never a last smaller batch with a different size and all batches are the same size (which we set to a
        # multiple of number of GPUs manually).
        mean_batch_loss = tf.cond(pred=tf.size(y_true) == 0,
                                  true_fn=lambda: 0.0,  # Avoid dividing by 0 if batch is empty.
                                  false_fn=lambda: tf.divide(total_loss, n_items))

        return mean_batch_loss


class MaskedMetric(tf.keras.metrics.Metric):

    def __init__(self, metric, name, **kwargs):
        super(MaskedMetric, self).__init__(name=name, **kwargs)
        self._metric = metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("Masked metrics do not support sample_weight.")

        mask = tf.logical_not(tf.equal(y_true, LABEL_PAD))
        self._metric.update_state(y_true, y_pred, sample_weight=mask)

    def result(self):
        return self._metric.result()

    def reset_states(self):
        self._metric.reset_states()


class BestModelSaverCallback(tf.keras.callbacks.Callback):
    def __init__(self, savedmodel_path):
        super(BestModelSaverCallback, self).__init__()
        self.savedmodel_path = savedmodel_path
        self.best_val_loss = math.inf

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < self.best_val_loss:
            # save_path = os.path.join(self.savedmodel_path, 'epoch_%03d' % (epoch + 1))  # epoch is 0-based
            tf.saved_model.save(self.model, self.savedmodel_path, signatures={'serving_default': self.model.model_server})
            self.best_val_loss = logs['val_loss']
