import tensorflow as tf
import math


def load_vocabulary(vocab_file):
    if tf.io.gfile.isdir(vocab_file):
        # Strangely enough GFile does not raise an error when it is given a directory to read from.
        # Reported this on Github: https://github.com/tensorflow/tensorflow/issues/46282#issue-782000566
        raise IsADirectoryError(f'{vocab_file} is a directory.')

    with tf.io.gfile.GFile(vocab_file, 'r') as f:
        return tf.strings.strip(f.readlines())


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


class CustomExponentialDecayLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, limiting_learning_rate, decay_steps, decay_rate):
        super(CustomExponentialDecayLR, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.limiting_learning_rate = limiting_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def get_config(self):
        config = {
            'init_lr': self.initial_learning_rate,
            'limit_lr': self.limiting_learning_rate,
            'decay_steps': self.decay_steps,
            'decay_rate': self.decay_rate
        }
        return config

    def __call__(self, step):
        learning_rate = (self.initial_learning_rate - self.limiting_learning_rate) * tf.math.pow(self.decay_rate, (tf.math.divide(step, self.decay_steps))) + self.limiting_learning_rate
        return learning_rate


class BestModelSaverCallback(tf.keras.callbacks.Callback):
    def __init__(self, savedmodel_path):
        super(BestModelSaverCallback, self).__init__()
        self.savedmodel_path = savedmodel_path
        self.best_val_loss = math.inf

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < self.best_val_loss:
            # save_path = os.path.join(self.savedmodel_path, 'epoch_%03d' % (epoch + 1))  # epoch is 0-based
            call = self.model.call.get_concrete_function(self.model.get_serving_signature())
            tf.saved_model.save(self.model, self.savedmodel_path, signatures=call)
            # tf.saved_model.save(self.model, self.savedmodel_path, signatures={'serving_default': self.model.model_server})
            self.best_val_loss = logs['val_loss']


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


if __name__ == '__main__':
    lr = CustomLRSchedule(d_model=20)
    q = lr(tf.range(40000, dtype=tf.float32))
    from matplotlib import pyplot as plt
    plt.plot(q)
    plt.show()
