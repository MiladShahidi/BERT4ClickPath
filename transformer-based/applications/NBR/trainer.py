from input_pipeline import create_tf_dataset
# from sequence_transformer.training_utils import PositiveRate, PredictedPositives, MaskedF1, MaskedMetric, MaskedBinaryCrossEntropy
from sequence_transformer.metrics import BestModelSaverCallback, CustomLRSchedule, F1Score, PredictedPositives
from data_generator import ClickStreamGenerator
from sequence_transformer.clickstream_model import ClickstreamModel
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
import contextlib
from sequence_transformer.utils import load_vocabulary
from sequence_transformer.head import MultiLabel_MultiClass_classification, SoftMaxHead


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
    # training_files = os.path.join(is_training, 'training_data/part*')
    training_dataset = create_tf_dataset(source=training,
                                         training=True,
                                         batch_size=kwargs['batch_size'])

    # validation_files = os.path.join(validation, 'part*')
    validation_dataset = create_tf_dataset(source=validation,
                                           training=False,
                                           batch_size=kwargs['batch_size'])

    return training_dataset, validation_dataset


def create_input_test(test, **kwargs):
    """
    Args:
        test:
        **kwargs:
            Allowed kwargs are:
                batch_size

    Returns:

    """
    # training_files = os.path.join(is_training, 'training_data/part*')
    test_dataset = create_tf_dataset(source=test,
                                         training=False,
                                         batch_size=kwargs['batch_size'])

    return test_dataset

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
          training,
          validation,
          model_dir,
          gpu_count,
          ckpt_dir=None,
          **training_params):


    metrics = [F1Score()]

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

    # loss = MaskedBinaryCrossEntropy()
    # loss = tf.keras.backend.binary_crossentropy
    loss = tf.keras.losses.BinaryCrossentropy()

    with get_distribution_context(gpu_count):
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # # # Training callbacks

    # Reduce LR on Plateau
    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10)

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
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    callbacks = [tensorboard, save_checkpoint, early_stopping, reduce_lr_on_plateau]

    model.fit(training_dataset,
              epochs=training_params['n_epochs'],
              steps_per_epoch=training_params['steps_per_epoch'],
              validation_data=validation_dataset,
              validation_steps=training_params.get('validation_steps', None),  # if not provided, validate on all data
              callbacks=callbacks,
              verbose=1)

    return model


if __name__ == '__main__':
    timestamp = time.strftime('%b-%d_%H-%M-%S')  # Avoid overwriting files with same epoch number from older runs
    model_dir_root = os.path.join('../output', f'run_{timestamp}')
    saved_model_dir = os.path.join(model_dir_root, 'savedmodel')
    ckpt_dir = os.path.join(model_dir_root, 'ckpts')

    simulated_data = True
    if simulated_data:
        N_ITEMS = 100
        item_vocab_dir = 'data/simulated/vocabs'
        data_src = ClickStreamGenerator(n_items=N_ITEMS, write_vocab_files=True, vocab_dir=item_vocab_dir)
        item_vocab_path = os.path.join(item_vocab_dir, 'item_vocab.txt')
        output_vocab_size = len(load_vocabulary(item_vocab_path))
        training_data_src = data_src
        validation_data_src = data_src
        test_data_src = data_src
    else:
        root_data_dir = 'data'
        training_data_files = os.path.join(root_data_dir, 'train/*.tfrecord')
        validation_data_files = os.path.join(root_data_dir, 'validation/*.tfrecord')
        item_vocab_path = os.path.join(root_data_dir, 'vocabs/item_vocab.txt')
        output_vocab_size = len(load_vocabulary(item_vocab_path))

        training_data_src = training_data_files
        validation_data_src = training_data_files

    training_params = {
        'n_epochs': 1000,
        'steps_per_epoch': 10,
        'validation_steps': 10,
        'lr_warmup_steps': 4000,
        'lr': 1e-2,
        'lr_scale': 1e-1,
        'batch_size': 1000,
        'max_sess_len': 200,
        'ckpt_dir': ckpt_dir,
    }

    test_params = {
        'n_epochs': 1,
        'steps_per_epoch': 10,
        'validation_steps': 10,
        'lr_warmup_steps': 4000,
        'lr': 1e-2,
        'lr_scale': 1,
        'batch_size': 1,
        'max_sess_len': 200,
        'ckpt_dir': ckpt_dir,
    }

    final_layers_dims = [1024, 512, 256]
    head_unit = MultiLabel_MultiClass_classification(dense_layer_dims=final_layers_dims, output_vocab_size=output_vocab_size)
    # head_unit = SoftMaxHead(dense_layer_dims=final_layers_dims, output_vocab_size=output_vocab_size)



    model_params = {
        'num_encoder_layers': 1,
        'num_attention_heads': 1,
        'dropout_rate': 0.1,
        'head_unit': head_unit
    }

    input_config = {
        'sequential_input_config': {
            'items': ['feature1',
                      'feature2',
                      'feature3',
                      'feature4',
                      'feature5',
                      'feature6',
                      'feature7',
                      'feature8',
                      'feature9',
                      'feature10'],
        },
        'feature_vocabs': {
            'items': 'data/vocabs/item_vocab.txt',
        },
        'embedding_dims': {
            'items': 30,
            # 'events': 2
        },
        'segment_to_head': 0
    }



    # data_src = ReturnsDataGen(n_items=N_ITEMS, n_events=10, session_cohesiveness=COHESION, positive_rate=0.5,
    #                           write_vocab_files=True, vocab_dir='../data/vocabs')

    config = {**model_params, **input_config}
    model = create_model(gpu_count=0,
                         # ckpt_dir=training_params['ckpt_dir'],
                         **config)

    training_dataset, validation_dataset = create_input(
        training=training_data_src,
        validation=validation_data_src,
        **training_params  # batch_size and max_sess_len
    )

    test_dataset = create_input_test(test=test_data_src, **test_params)

    trained_model = train(model=model,
                          training=training_dataset,
                          validation=validation_dataset,
                          model_dir=saved_model_dir,
                          gpu_count=0,
                          **training_params)

    for x, y in test_dataset.take(3):
        print(x['feature1'])
        y_hat = trained_model.predict(x)
        print('y_hat')
        print(y_hat)
        print('Label:')
        print(y)
        print('*'*80)
        print('*'*80)
