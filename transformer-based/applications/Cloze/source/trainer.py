from source.input_pipeline import create_tf_dataset
from sequence_transformer.training_utils import PositiveRate, PredictedPositives, MaskedMetric, MaskedLoss
from sequence_transformer.training_utils import BestModelSaverCallback, CustomLRSchedule
from source.data_generator import ClickStreamGenerator
from sequence_transformer.clickstream_model import ClickstreamModel
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
import contextlib
from sequence_transformer.constants import INPUT_MASKING_TOKEN, LABEL_PAD, CLASSIFICATION_TOKEN
from sequence_transformer.head import SoftMaxHead
from sequence_transformer.utils import load_vocabulary, parse_cmd_line_arguments


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
    training_dataset = create_tf_dataset(source=training,
                                         is_training=True,
                                         batch_size=kwargs['batch_size'],
                                         target_vocab_file=kwargs['target_vocab_file'])

    validation_dataset = create_tf_dataset(source=validation,
                                           is_training=False,
                                           batch_size=kwargs['batch_size'],
                                           target_vocab_file=kwargs['target_vocab_file'])

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
        model = ClickstreamModel(**config)

    if ckpt_dir is not None:
        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if latest_ckpt is not None:
            print(f'Loading all layer weights from {latest_ckpt}')
            model.load_weights(latest_ckpt)
        else:
            print(f'Warning: No checkpoint found in {ckpt_dir}')

    return model


def train(model,
          training_data,
          validation_data,
          model_dir,
          gpu_count,
          ckpt_dir=None,
          **training_params):

    metrics = [
        # PositiveRate(),
        # PredictedPositives(),
        # MaskedMetric(metric=tf.keras.metrics.Recall(), name='recall'),
        # MaskedMetric(metric=tf.keras.metrics.Precision(), name='precision'),
        MaskedMetric(metric=tf.keras.metrics.SparseCategoricalAccuracy(), name='Accuracy'),
        # MaskedMetric(metric=tf.keras.metrics.AUC(curve='PR'), name='prauc'),
        # MaskedMetric(metric=tf.keras.metrics.AUC(curve='ROC'), name='rocauc'),
        # MaskedMetric(metric=tf.keras.metrics.PrecisionAtRecall(recall=0.1), name='precision-at-10'),
        # MaskedMetric(metric=tf.keras.metrics.PrecisionAtRecall(recall=0.2), name='precision-at-20'),
        # MaskedMetric(metric=tf.keras.metrics.PrecisionAtRecall(recall=0.3), name='precision-at-30')
    ]

    d_model = sum(model.embedding_dims.values())

    # lr = CustomLRSchedule(d_model=d_model, scale=training_params['lr_scale'])
    # lr = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=training_params['lr'],
    #     decay_steps=training_params['steps_per_epoch'],
    #     decay_rate=0.9,
    #     staircase=True
    # )
    lr = training_params['lr']
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss = MaskedLoss(tf.keras.backend.sparse_categorical_crossentropy, label_pad=tf.cast(LABEL_PAD, tf.int64))
    # loss = tf.keras.losses.SparseCategoricalCrossentropy()

    with get_distribution_context(gpu_count):
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # # # Training callbacks
    callbacks = []

    # Reduce LR on Plateau
    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10)

    if ckpt_dir is not None:
        # Checkpoint saver
        ckpt_timestamp = time.strftime('%b-%d_%H-%M-%S')  # Avoid overwriting files with same epoch number from older runs
        ckpt_filename = 'ckpt-' + ckpt_timestamp + '-{epoch:04d}.ckpt'  # will include epoch in filename
        checkpoint_path = os.path.join(ckpt_dir, ckpt_filename)
        save_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=False, verbose=1)
        callbacks += [save_checkpoint]

    # Tensorboard
    tensorboard_path = os.path.join(model_dir, 'tensorboard')
    # TODO: Adding embeddings_freq to this callback results in the following error:
    #  AttributeError: Embedding object has no attribute embeddings
    tensorboard = TensorBoard(log_dir=tensorboard_path, profile_batch=0)  # , embeddings_freq=5)

    # Model Saver
    saved_model_dir = os.path.join(model_dir, 'savedmodel')
    best_model_saver = BestModelSaverCallback(savedmodel_path=saved_model_dir)

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    callbacks += [tensorboard, early_stopping, reduce_lr_on_plateau]

    model.fit(training_data,
              epochs=training_params['n_epochs'],
              steps_per_epoch=training_params['steps_per_epoch'],
              validation_data=validation_data,
              validation_steps=training_params.get('validation_steps', None),  # if not provided, validate on all data
              callbacks=callbacks,
              verbose=1)

    return model


if __name__ == '__main__':
    timestamp = time.strftime('%b-%d_%H-%M-%S')  # Avoid overwriting files with same epoch number from older runs
    model_dir = os.path.join('../training_output', f'run_{timestamp}')
    root_data_dir = '../data/amazon_beauty_bert4rec'

    # simulated_data = False
    # if simulated_data:
    #     N_ITEMS = 100
    #     COHESION = 100
    #     item_vocab_dir = 'data/simulated/vocabs'
    #     data_src = ClickStreamGenerator(n_items=N_ITEMS, n_events=10, session_cohesiveness=COHESION,
    #                                     write_vocab_files=True, vocab_dir=item_vocab_dir)
    #     item_vocab_file = os.path.join(item_vocab_dir, 'item_vocab.txt')
    #     output_vocab_size = len(load_vocabulary(item_vocab_file))
    #     training_data_src = data_src
    #     validation_data_src = data_src
    # else:

    training_param_spec = {
        'input_data': root_data_dir,
        'model_dir': model_dir,
        'n_epochs': 10000,
        'steps_per_epoch': 1000,
        'validation_steps': 100,
        'lr_warmup_steps': 4000,
        'lr': 1e-3,
        'lr_scale': 1.0,
        'batch_size': 500,
        'max_sess_len': 200,
        'job-dir': 'placeholder'
    }

    model_param_spec = {
        'num_encoder_layers': 4,
        'num_attention_heads': 4,
        'dropout_rate': 0.1,
    }

    parsed_args = parse_cmd_line_arguments({**training_param_spec, **model_param_spec})
    model_params = {k: parsed_args.get(k, None) for k in model_param_spec.keys()}
    training_params = {k: parsed_args.get(k, None) for k in training_param_spec.keys()}
    training_params.pop('job-dir')

    from pprint import pprint
    print('*'*80)
    pprint(model_params)
    pprint(training_params)
    print('*'*80)

    training_data_files = os.path.join(training_params['input_data'], '*.tfrecord')
    validation_data_files = training_data_files

    item_vocab_file = os.path.join(training_params['input_data'], 'vocabs/item_vocab.txt')
    output_vocab_size = len(load_vocabulary(item_vocab_file))

    # saved_model_dir = os.path.join(training_params['model_dir'], 'savedmodel')
    ckpt_dir = os.path.join(training_params['model_dir'], 'ckpts')

    input_config = {
        'sequential_input_config': {
            'items': ['asin'],
            # 'events': ['seq_1_events', 'seq_2_events']
        },
        'feature_vocabs': {
            'items': item_vocab_file,
            # 'events':
        },
        'embedding_dims': {
            'items': 40,
            # 'events': 2
        },
        'value_to_head': INPUT_MASKING_TOKEN
        # 'segment_to_head': 0
    }

    config = {**model_params, **input_config}

    final_layers_dims = [1024, 512, 256]
    head_unit = SoftMaxHead(dense_layer_dims=final_layers_dims, output_vocab_size=output_vocab_size)

    model = create_model(gpu_count=1,
                         # ckpt_dir=training_params['ckpt_dir'],
                         **config,
                         head_unit=head_unit)

    training_dataset, validation_dataset = create_input(
        training=training_data_files,
        validation=validation_data_files,
        target_vocab_file=item_vocab_file,
        **training_params  # batch_size and max_sess_len
    )

    do_train = True
    if do_train:
        model = train(model=model,
                      training_data=training_dataset,
                      validation_data=validation_dataset,
                      gpu_count=1,
                      **training_params)

    # import numpy as np
    # import sys
    # np.set_printoptions(threshold=sys.maxsize, precision=2)
    #
    # for x, y in validation_dataset.take(1):
    #     print(x['asin'])
    #     print('*'*80)
    #     print(y)
    #     # y_hat = model(x)
    #     # print(head_inp)
    #     # print(y_hat.numpy())
    #     # print(np.argmax(y_hat.numpy(), axis=-1))
    #     # emb = model.transformer.embedding_layers['items']
    #     # print(emb.input_dim)
    #     # print(emb.output_dim)
    #     # print(emb(tf.convert_to_tensor(range(N_ITEMS+11))))
    #
    # # tf.keras.backend.set_learning_phase(0)
    # # tf.saved_model.save(trained_model, saved_model_dir, signatures={'serving_default': trained_model.model_server})
