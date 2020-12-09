import tensorflow as tf
from trainer import create_input
from training_utils import MaskedMetric, MaskedFocalLoss
import os
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
from scipy.signal import savgol_filter


def reduce_basket(y, threshold):
    return np.any(np.where(np.greater_equal(y, threshold), 1, 0), axis=1).astype(np.int)


def evaluate(training,
             test,
             saved_model_dir):

    saved_model_file = [filename for filename in os.listdir(saved_model_dir) if filename.endswith('.pb')][0]
    print(f'Loading {os.path.join(saved_model_dir, saved_model_file)}')
    model = tf.keras.models.load_model(
        filepath=saved_model_dir,
        compile=False
    )

    training_dataset, test_dataset = create_input(
        training=training,
        validation=test,
        batch_size=500,
        max_sess_len=200,
        prediction_level='item'
    )

    model.compile(loss=MaskedFocalLoss())

    basket_level = False
    y_true = []
    y_pred = []
    n_steps = 50
    for i, (x, y) in enumerate(test_dataset.take(n_steps)):
        t1 = datetime.now()
        live_model_features = [
            'basket_product_id',
            'discount',
            'ordered_quantity',
            'unit_price',
            'shipping_charge',
            'event_name',
            'page_type',
            'product_skn_id_events',
            'product_skn_id_page_views'
        ]
        x = {k: x[k] for k in live_model_features}
        predict = model.signatures["serving_default"]

        batch_y_pred = predict(**x)['scores'].numpy()
        batch_y_true = y.numpy()

        mask = np.not_equal(batch_y_true, -1)

        if basket_level:
            batch_y_pred = batch_y_pred * mask
            batch_y_pred = np.max(batch_y_pred, axis=1)
            batch_y_true = np.max(batch_y_true, axis=1)
        else:
            mask = mask.ravel()
            batch_y_pred = batch_y_pred.ravel()[mask]
            batch_y_true = batch_y_true.ravel()[mask]

        y_true.extend(batch_y_true)
        y_pred.extend(batch_y_pred)

        print('%03d out of %d done in %1.2f seconds.' % (i+1, n_steps, (datetime.now() - t1).total_seconds()))

    pr, rc, th = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
    plt.plot(th, pr[:-1])
    plt.plot([0, 1], [0, 1])
    # smooth_pr = savgol_filter(pr, 51, 3)
    # plt.plot(rc[:-1], smooth_pr[:-1])
    # plt.scatter([0, 1], c='white')  # Force y axis to [0, 1]
    plt.show()
    plt.hist(y_pred, bins=100)
    plt.show()


if __name__ == '__main__':
    saved_model_dir = '../../output/live/1'

    training = os.path.join('../data/mock_v5/training')
    validation = os.path.join('../data/mock_v5/validation_test_nov/test_nov_random')

    evaluate(training=training,
             test=validation,
             saved_model_dir=saved_model_dir)
