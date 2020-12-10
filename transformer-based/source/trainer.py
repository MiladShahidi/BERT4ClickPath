from input_pipeline import create_tf_dataset
from training_utils import PositiveRate, PredictedPositives, MaskedF1, MaskedMetric, MaskedBinaryCrossEntropy
from training_utils import BestModelSaverCallback, CustomLRSchedule
from data_generator import ReturnsDataGen
from returns_model import ReturnsModel
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
import contextlib


def get_vocab_files(training):
    item_vocab_path = os.path.join(training, 'vocabs/skn_id_vocab')
    event_vocab_path = os.path.join(training, 'vocabs/page_type_event_name_vocab')
    item_vocab_file = [filename for filename in tf.io.gfile.glob(os.path.join(item_vocab_path, '*'))
                       if (not tf.io.gfile.isdir(filename)) and filename.endswith('.txt')][0]  # Expects only 1 txt file
    event_vocab_file = [filename for filename in tf.io.gfile.glob(os.path.join(event_vocab_path, '*'))
                        if (not tf.io.gfile.isdir(filename)) and filename.endswith('.txt')][0]

    return {'item_vocab_file': item_vocab_file, 'event_vocab_file': event_vocab_file}


def create_input(training, validation, **kwargs):
    """
    Args:
        training:
        validation:
        **kwargs:
            Allowed kwargs are:
                batch_size
                max_sess_len

    Returns:

    """
    # training_files = os.path.join(training, 'training_data/part*')
    training_dataset = create_tf_dataset(source=training,
                                         training=True,
                                         batch_size=kwargs['batch_size'])

    # validation_files = os.path.join(validation, 'part*')
    validation_dataset = create_tf_dataset(source=validation,
                                           training=False,
                                           batch_size=kwargs['batch_size'])

    return training_dataset, validation_dataset


def get_distribution_context(gpu_count):
    if gpu_count > 1:
        strategy = tf.distribute.MirroredStrategy()
        dist_context = strategy.scope()
    else:
        dist_context = contextlib.suppress()  # A no-op context

    return dist_context


def create_model(gpu_count, ckpt_dir=None, **config):
    """
    Args:
        ckpt_dir:

    Returns:

    """
    with get_distribution_context(gpu_count):
        model = ReturnsModel(**config)

    if ckpt_dir is not None:
        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if latest_ckpt is not None:
            print(f'Loading all layer weights from {latest_ckpt}')
            model.load_weights(latest_ckpt)
        else:
            print(f'Warning: No checkpoint found in {ckpt_dir}')

    return model


def train(model,
          training,
          validation,
          model_dir,
          gpu_count,
          ckpt_dir=None,
          **training_params):

    # train_window = (model_params['train_from'], model_params['train_until'])
    # validation_window = (model_params['validate_from'], model_params['validate_until'])
    training_dataset, validation_dataset = create_input(
        training=training,
        validation=validation,
        **training_params  # batch_size and max_sess_len
    )

    metrics = [
        PositiveRate(),
        PredictedPositives(),
        MaskedF1(),
        MaskedMetric(metric=tf.keras.metrics.Recall(), name='recall'),
        MaskedMetric(metric=tf.keras.metrics.Precision(), name='precision'),
        # MaskedMetric(metric=tf.keras.metrics.AUC(curve='PR'), name='prauc'),
        # MaskedMetric(metric=tf.keras.metrics.AUC(curve='ROC'), name='rocauc'),
        # MaskedMetric(metric=tf.keras.metrics.PrecisionAtRecall(recall=0.1), name='precision-at-10'),
        # MaskedMetric(metric=tf.keras.metrics.PrecisionAtRecall(recall=0.2), name='precision-at-20'),
        # MaskedMetric(metric=tf.keras.metrics.PrecisionAtRecall(recall=0.3), name='precision-at-30')
    ]

    d_model = sum(model.embedding_dims.values())

    lr = CustomLRSchedule(d_model=d_model, scale=training_params['lr_scale'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss = MaskedBinaryCrossEntropy()

    with get_distribution_context(gpu_count):
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # # # Training callbacks

    # Checkpoint saver
    ckpt_timestamp = time.strftime('%b-%d_%H-%M-%S')  # Avoid overwriting files with same epoch number from older runs
    ckpt_filename = 'ckpt-' + ckpt_timestamp + '-{epoch:04d}.ckpt'  # will include epoch in filename
    checkpoint_path = os.path.join(ckpt_dir, ckpt_filename)
    save_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=False, verbose=1)

    # Tensorboard
    tensorboard_path = os.path.join(ckpt_dir, 'tensorboard')
    # TODO: Adding embeddings_freq to this callback results in the following error:
    #  AttributeError: Embedding object has no attribute embeddings
    tensorboard = TensorBoard(log_dir=tensorboard_path, profile_batch=0)  # , embeddings_freq=5)

    # Model Saver
    best_model_saver = BestModelSaverCallback(savedmodel_path=model_dir)
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    callbacks = [tensorboard, save_checkpoint, early_stopping]

    model.fit(training_dataset,
              epochs=training_params['n_epochs'],
              steps_per_epoch=training_params['steps_per_epoch'],
              validation_data=validation_dataset,
              validation_steps=training_params.get('validation_steps', None),  # if not provided, validate on all data
              callbacks=callbacks,
              verbose=1)

    return model


if __name__ == '__main__':
    model_dir_root = '../output'
    saved_model_dir = os.path.join(model_dir_root, 'savedmodel')
    ckpt_dir = os.path.join(model_dir_root, 'ckpts')

    training_params = {
        'n_epochs': 5,
        'steps_per_epoch': 100,
        'validation_steps': 100,
        'lr_warmup_steps': 4000,
        'lr': 1e-3,
        'lr_scale': 0.2,
        'batch_size': 100,
        'max_sess_len': 200,
        'ckpt_dir': ckpt_dir,
    }

    model_params = {
        'num_encoder_layers': 1,
        'num_attention_heads': 1,
        'dropout_rate': 0.35,
        'final_layers_dims': [256, 128]
    }

    input_config = {
        'input_seq_mapping': {
            'items': ['seq_1_items', 'seq_2_items'],
            'events': ['seq_1_events', 'seq_2_events']
        },
        'feature_vocabs': {
            'items': '../data/vocabs/item_vocab.txt',
            'events': '../data/vocabs/event_vocab.txt'
        },
        'embedding_dims': {
            'items': 4,
            'events': 2
        }
    }

    training_data_src = ReturnsDataGen(n_items=10, n_events=10)
    validation_data_src = ReturnsDataGen(n_items=10, n_events=10)

    config = {**model_params, **input_config}
    model = create_model(gpu_count=0,
                         # ckpt_dir=training_params['ckpt_dir'],
                         **config)

    trained_model = train(model=model,
                          training=training_data_src,
                          validation=validation_data_src,
                          model_dir=saved_model_dir,
                          gpu_count=0,
                          **training_params)

    # tf.keras.backend.set_learning_phase(0)
    # tf.saved_model.save(trained_model, saved_model_dir, signatures={'serving_default': trained_model.model_server})
