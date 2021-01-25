import tensorflow as tf
import math
from sequence_transformer.constants import LABEL_PAD


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
