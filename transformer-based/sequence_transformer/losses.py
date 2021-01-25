import tensorflow as tf
from sequence_transformer.constants import LABEL_PAD


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
        item_loss = self.item_wise_loss_fn(y_true, y_pred)
        # Some functions squeeze the resulting tensor but we need this to have the same shape as y_true
        # Otherwise it will get messed up (wrong dimensions) when multiplied by mask below
        item_loss = tf.reshape(item_loss, tf.shape(y_true))

        # # # # # # # # # # #
        # # # Mask padded items
        # # # # # # # # # # #
        mask = tf.cast(mask, dtype=item_loss.dtype)
        item_loss *= mask  # Losses corresponding to padded positions are multiplied by 0 (masked)

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


if __name__ == '__main__':
    y_true = [[1, -1], [2, -1]]
    y_pred = [
        [
            [0.9, 0.05, 0.05],
            [0.5, 0.3, 0.2]
        ],
        [
            [0.9, 0.05, 0.05],
            [0.5, 0.3, 0.2]
        ]
    ]
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    loss_fn = ClozeMaskedLoss(tf.keras.backend.sparse_categorical_crossentropy, label_pad=tf.cast(LABEL_PAD, tf.int32))
    l_1 = loss_fn(y_true, y_pred)
    print('Cloze:', l_1)
    print('*' * 80)

    masked_loss_fn = MaskedLoss(tf.keras.backend.sparse_categorical_crossentropy, label_pad=tf.cast(LABEL_PAD, tf.int32))
    l_2 = masked_loss_fn(y_true, y_pred)
    print('Masked:', l_2)
    print('*' * 80)

    y_true = [[1, 2]]
    y_pred = [[
        [0.9, 0.05, 0.05],
        # [0.5, 0.3, 0.2],
        [0.9, 0.05, 0.05]
    ]]
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    keras_loss_fn = tf.keras.backend.sparse_categorical_crossentropy
    # print(tf.reduce_mean(keras_loss_fn(y_true, y_pred)))
