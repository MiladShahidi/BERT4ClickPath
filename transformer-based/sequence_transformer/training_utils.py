import tensorflow as tf
from sequence_transformer.constants import LABEL_PAD
import math
# from focal_loss import SigmoidFocalCrossEntropy


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


class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self, item_wise_loss_fn, pos_weight=None, label_pad=LABEL_PAD):
        """

        Args:
            item_wise_loss_fn: An item-wise loss function that does not perform any reduction (e.g. mean over batch).
                For example, `tf.keras.backend.binary_crossentropy` is one such loss function.
                NOTE: usual loss functions like `tf.keras.losses.binary_crossentropy` performs reduction
            pos_weight: weight for positive class for binary labels. Only works as expected for binary labels.
        """
        super(MaskedLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE)
        self.item_wise_loss_fn = item_wise_loss_fn

        assert label_pad < 0, "label_pad must be less than zero, to distinguish it from actual labels."
        self.label_pad = label_pad

        if pos_weight is not None:
            print('*'*80)
            print('WARNING: providing pos_weight to a masked loss only works as expected for binary labels.')
            print('*'*80)

        self.pos_weight = tf.cast(pos_weight, tf.float32) if pos_weight is not None else None
        self._negative_weight = 1.0  # This is only to make the code more readable and possibly easier to extend later

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: labels, possibly padded (usually using -1 as the pad).
            y_pred: logits from the model

        Returns:
        """
        # Note: I'm using the Keras backend function (below) instead of tf.keras.losses.BinaryCrossentropy, or
        # tf.keras.losses.binary_crossentropy, because these return the mean loss per example, while
        # I need individual losses for output nodes because I need to mask the padded ones before averaging
        # them. (additional note: the reduction argument above prevents averaging across batches, not per node)

        # Note this is backend and not tf.keras.losses.binary_crossentropy. The difference is that this one does not
        # perform any reduction while the one in the losses module, even when reduction=None, reduces the last dimension

        # # # # # # # # # # #
        # # # Create the mask
        # # # # # # # # # # #
        self.label_pad = tf.cast(self.label_pad, y_true.dtype)
        mask = tf.math.logical_not(tf.math.equal(y_true, self.label_pad))
        mask = tf.cast(mask, y_true.dtype)  # Convert Boolean tensor to numerical

        # # # # # # # # # # #
        # # # Calculate item-level loss
        # # # # # # # # # # #
        # The loss function may not be able to accept the pad value (especially that pad is < 0). So we will replace
        # label pads with 0s. Note that these will be masked in the next step and won't get mixed with actual labels.
        pad_offset = (1 - mask) * self.label_pad  # This will have label_pad in padded positions and 0 everywhere else
        y_true -= pad_offset  # replaces label pads with 0s. Leaves all other labels intact.

        # (batch_size, input_seq_len)
        item_loss = self.item_wise_loss_fn(y_true, y_pred, from_logits=False)

        # # # # # # # # # # #
        # # # Mask padded items
        # # # # # # # # # # #
        mask = tf.cast(mask, dtype=item_loss.dtype)
        item_loss *= mask  # Loses corresponding to padded positions are multiplied by 0 (masked)

        if self.pos_weight is not None:
            sample_weight = tf.where(tf.equal(y_true, 1), self.pos_weight, self._negative_weight)
            item_loss = tf.multiply(sample_weight, item_loss)

        # # # # # # # # # # #
        # # # Reduction
        # # # # # # # # # # #
        # Note: Gives all items equal weight. i.e. the loss from the only item of a 1-label example is given the same
        # weight as that of each item in a 2-label example. So we just average over all items in the batch
        total_loss = tf.reduce_sum(item_loss)
        n_items = tf.reduce_sum(mask)

        # In a multi-GPU setup, each global batch will be split between GPUs and if batch size varies, or is not a
        # multiple of number of GPUs, some GPUs will receive empty batches. This can cause nan loss. One work-around
        # is to explicitly handle this case here.
        # https://stackoverflow.com/questions/54283937/training-on-multiple-gpus-causes-nan-validation-errors-in-keras
        # This does not happen during is_training. My guess is that because is_training data is repeated indefinitely,
        # there is never a last smaller batch with a different size and all batches are the same size (which we set to a
        # multiple of number of GPUs manually).
        mean_batch_loss = tf.cond(pred=tf.size(y_true) == 0,
                                  true_fn=lambda: 0.0,  # Avoid dividing by 0 if batch is empty.
                                  false_fn=lambda: tf.divide(total_loss, n_items))

        # # # Normalize for weights. Optional. But it will keep the loss on the same scale as unweighted loss
        if self.pos_weight is not None:
            weight_normalization = (self.pos_weight + self._negative_weight) / 2
            mean_batch_loss = mean_batch_loss / weight_normalization

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
        # This does not happen during is_training. My guess is that because is_training data is repeated indefinitely, there
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


if __name__ == '__main__':
    import numpy as np

    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    N_CLASSES = 4
    BATCH = 5
    N_MASKED = 10
    VOCAB_SIZE = 20

    y_true_1 = [1, 2]
    y_pred_1 = [[0.1, 0.8, 0.1], [0.9, 0.0, 0.1]]
    l_1 = loss(
        y_true_1,
        y_pred_1
    )

    y_true_2 = [0]
    y_pred_2 = [[0.1, 0.8, 0.1]]
    l_2 = loss(
        y_true_2,
        y_pred_2
    )

    print(l_1)
    print(l_2)
    print(np.mean([l_1, l_2]))

    print('*'*80)

    y_true_2 = [0, -1]
    y_pred_2 = [[0.1, 0.8, 0.1], [0, 0, 0]]

    y_true = np.array([y_true_1, y_true_2])
    y_pred = np.array([y_pred_1, y_pred_2])

    print(y_true)
    print(y_pred)

    masked_loss = MaskedLoss(tf.keras.backend.sparse_categorical_crossentropy, label_pad=tf.cast(LABEL_PAD, tf.int64))

    l = masked_loss(
        y_true,
        y_pred
    )

    print(l)
