import json
from source.input_pipeline import create_cloze_dataset
from clickstream_transformer.metrics import PositiveRate, PredictedPositives, MaskedMetric
from clickstream_transformer.training_utils import BestModelSaverCallback, CustomLRSchedule, LRTensorBoard
from clickstream_transformer.losses import MaskedLoss
from source.utils import ClozeMaskedLoss, ClozeMaskedNDCG, ClozeMaskedRecall, parse_cmd_line_arguments
from source.cloze_constants import modes
from source.data_generator import ClickStreamGenerator
from clickstream_transformer.clickstream_transformer import ClickstreamTransformer
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
import contextlib
from clickstream_transformer.constants import INPUT_MASKING_TOKEN, LABEL_PAD
from clickstream_transformer.head import SoftMaxHead
from clickstream_transformer.training_utils import load_vocabulary


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
    training_dataset = create_cloze_dataset(source=training,
                                            mode=modes.TRAIN,
                                            batch_size=kwargs['batch_size'],
                                            target_vocab_file=kwargs['target_vocab_file'])

    validation_dataset = create_cloze_dataset(source=validation,
                                              mode=modes.EVAL,
                                              batch_size=kwargs['batch_size'],
                                              target_vocab_file=kwargs['target_vocab_file'])

    return training_dataset, validation_dataset


def get_distribution_context(gpu_count):
    # I think this should only be called once, because every time it is called it returns a new distribution strategy
    # object and using different objects at different stages (creating the model, compiling, etc.) will cause an error.
    # RuntimeError: Mixing different tf.distribute.Strategy objects:
    # <[...]MirroredStrategy object at 0x7f2e16e60410> is not <[...]MirroredStrategy object at 0x7f2>
    if gpu_count > 1:
        strategy = tf.distribute.MirroredStrategy()
        dist_context = strategy.scope()
    else:
        dist_context = contextlib.suppress()  # A no-op context

    return dist_context


def get_training_spec(training_params):
    # This function is only used to hide away these details from the main body of code
    metrics = [
        # PositiveRate(),
        # PredictedPositives(),
        # MaskedMetric(metric=tf.keras.metrics.Recall(), name='recall'),
        # MaskedMetric(metric=tf.keras.metrics.Precision(), name='precision'),
        # MaskedMetric(metric=tf.keras.metrics.SparseCategoricalAccuracy(), name='Accuracy'),
        ClozeMaskedNDCG(k=5),
        ClozeMaskedNDCG(k=10),
        # ClozeMaskedRecall(k=5),
        # ClozeMaskedRecall(k=10)
        # MaskedMetric(metric=tf.keras.metrics.AUC(curve='PR'), name='prauc'),
        # MaskedMetric(metric=tf.keras.metrics.AUC(curve='ROC'), name='rocauc'),
        # MaskedMetric(metric=tf.keras.metrics.PrecisionAtRecall(recall=0.1), name='precision-at-10'),
        # MaskedMetric(metric=tf.keras.metrics.PrecisionAtRecall(recall=0.2), name='precision-at-20'),
        # MaskedMetric(metric=tf.keras.metrics.PrecisionAtRecall(recall=0.3), name='precision-at-30')
    ]

    # lr = CustomLRSchedule(d_model=d_model, scale=training_params['lr_scale'])
    # lr = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=training_params['lr'],
    #     decay_steps=training_params['steps_per_epoch'],
    #     decay_rate=0.9,
    #     staircase=True
    # )
    lr = training_params['lr']
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-9)

    loss = ClozeMaskedLoss(tf.keras.backend.sparse_categorical_crossentropy, label_pad=LABEL_PAD)
    # loss = MaskedLoss(tf.keras.backend.sparse_categorical_crossentropy, label_pad=tf.cast(LABEL_PAD, tf.int64))
    # loss = tf.keras.losses.SparseCategoricalCrossentropy()

    return {
        'metrics': metrics,
        'loss': loss,
        'optimizer': optimizer
    }


def create_model(ckpt_dir=None, **config):
    """

    Args:
        ckpt_dir: Loads model weights from the latest checkpoint found in this directory.
        **config:

    Returns:

    """
    model = ClickstreamTransformer(**config)

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
          # ckpt_dir=None,
          **training_params):

    # # # Training callbacks
    callbacks = []

    # Reduce LR on Plateau
    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.317)

    # Checkpoint saver
    ckpt_dir = os.path.join(model_dir, 'ckpts')
    ckpt_timestamp = time.strftime('%b-%d_%H-%M-%S')  # Avoid overwriting files with same epoch no. from older runs
    ckpt_filename = 'ckpt-' + ckpt_timestamp + '{epoch:04d}'  # will include epoch in filename
    checkpoint_path = os.path.join(ckpt_dir, ckpt_filename)
    save_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, verbose=2)
    callbacks += [save_checkpoint]

    # Tensorboard
    tensorboard_path = os.path.join(model_dir, 'tensorboard')

    # Adding embeddings_freq to this callback results in the following error:
    # AttributeError: Embedding object has no attribute embeddings
    tensorboard = TensorBoard(log_dir=tensorboard_path, profile_batch=0)  # , embeddings_freq=5)

    # Model Saver
    saved_model_dir = os.path.join(model_dir, 'savedmodel')
    best_model_saver = BestModelSaverCallback(savedmodel_path=saved_model_dir)

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    callbacks += [tensorboard, early_stopping, reduce_lr_on_plateau, best_model_saver]

    model.fit(training_data,
              epochs=training_params['n_epochs'],
              steps_per_epoch=training_params['steps_per_epoch'],
              validation_data=validation_data,
              validation_steps=training_params.get('validation_steps', None),  # if not provided, validate on all data
              callbacks=callbacks,
              verbose=1)

    return model


if __name__ == '__main__':
    TASK = 'train'  # or 'train'
    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    local_run = (tf_config == {})  # Trying to be specific and avoid logical not of a dict! (not tf_config)

    if not local_run:
        # For this to work the config .yaml file must specify master_config etc. explicitly, instead of using
        # pre-defined identifiers like complex_model_gpu
        N_PROCESSORS = int(tf_config['job']['master_config']['accelerator_config']['count'])  # Cloud GPUs
    else:
        N_PROCESSORS = 1  # local CPU

    timestamp = time.strftime('%d-%b-%H-%M-%S')  # Avoid overwriting files with same epoch number from older runs
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

    PER_GPU_BATCH_SIZE = 512
    LOCAL_BATCH_SIZE = 100
    CLOUD_BATCH_SIZE = PER_GPU_BATCH_SIZE * N_PROCESSORS  # In distributed training N_PROCESSORS > 1 (= no. of GPUs)

    # These are the default values. If provided through the command line (e.g. for hyperparameter tuning), these will
    # be replaced by the cmd line values
    training_param_spec = {
        'input_data': root_data_dir,
        'model_dir': model_dir,
        'n_epochs': 10000,
        'steps_per_epoch': 1000 if not local_run else 50,
        'validation_steps': 500 if not local_run else 100,
        'lr_warmup_steps': 4000,
        'lr': 5e-4,
        'lr_scale': 1.0,
        'batch_size': CLOUD_BATCH_SIZE if not local_run else LOCAL_BATCH_SIZE,
        'max_sess_len': 50,  # ToDo: Enable this. This is set to 50 in BERT4Rec for Amazon Beauty dataset
        'init_ckpt_dir': 'DUMMY',  # Provide from command line if needed
        'job-dir': 'placeholder'
    }

    model_param_spec = {
        'num_encoder_layers': 2,
        'num_attention_heads': 2,
        'dropout_rate': 0.1,
    }

    parsed_args = parse_cmd_line_arguments({**training_param_spec, **model_param_spec})
    model_params = {k: parsed_args.get(k, None) for k in model_param_spec.keys()}
    training_params = {k: parsed_args.get(k, None) for k in training_param_spec.keys()}
    training_params.pop('job-dir')

    data_files = os.path.join(training_params['input_data'], '*.tfrecord')

    item_vocab_file = os.path.join(training_params['input_data'], 'vocabs/item_vocab.txt')
    output_vocab_size = len(load_vocabulary(item_vocab_file))

    ckpt_dir = os.path.join(training_params['model_dir'], 'ckpts')

    d_model = 64

    input_config = {
        'sequential_input_config': {
            'items': ['asin'],
            # 'events': ['seq_1_events']
        },
        'feature_vocabs': {
            'items': item_vocab_file,
            # 'events': event_vocab_file
        },
        'embedding_dims': {
            'items': d_model,
            # 'events': 2
        },
        'value_to_head': INPUT_MASKING_TOKEN
        # 'segment_to_head': 0
    }

    config = {**model_params, **input_config}

    print('*'*80)
    print(training_params)
    print(config)
    print('*'*80)

    final_layers_dims = [1024, 512]
    head_unit = SoftMaxHead(dense_layer_dims=final_layers_dims, output_vocab_size=output_vocab_size)

    # source data files are the same for training and validation but input_pipeline will process them differently
    training_dataset, validation_dataset = create_input(
        training=data_files,
        validation=data_files,
        target_vocab_file=item_vocab_file,
        **training_params  # batch_size and max_sess_len
    )

    with get_distribution_context(N_PROCESSORS):
        model = create_model(ckpt_dir=training_params['init_ckpt_dir'],
                             head_unit=head_unit, **config)
        training_spec = get_training_spec(training_params)  # This must be created inside the distribution scope
        model.compile(**training_spec)

    # This will:
    # - train the model
    # - write metrics for Tensorboard
    # - save checkpoints regularly
    # - save the latest best model (according to validation loss)
    model = train(model=model,
                  training_data=training_dataset,
                  validation_data=validation_dataset,
                  **training_params)
