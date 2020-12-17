import os
import time
import logging
import torch
from utils.data_generator import ReturnsDataGen
from train import train_model, validate_model
from predict import batch_predict
from utils import data_helpers as dh
from config import Config
from rnn_model import DRModel
import warnings
warnings.filterwarnings("ignore")
logger = dh.logger_fn("torch-log", "logs/training.log")


class DreamModelTraining():
    def __init__(self, data_generator,  **kwargs):
        # Model configuration
        self.config = Config.from_dict(kwargs)
        self.config.batch_size = data_generator.batch_size
        self.config.num_product = data_generator.num_product
        self.config.seq_len = data_generator.seq_len
        self.kwargs = kwargs
        self.model_instance = DRModel(self.config)
        self.data_generator = data_generator
        self.best_epoch = 0

    def _do_training(self):

        # Optimizer
        optimizer = torch.optim.Adam(self.model_instance.parameters(), lr=self.config.learning_rate)

        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", str(int(time.time()))))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        logger.info('Save into {0}'.format(out_dir))
        checkpoint_dir = out_dir + '/model-{epoch:02d}.model'
        best_val_loss = None

        try:
            # Training
            for epoch in range(self.config.epochs):
                self.model_instance = train_model(self.model_instance, self.data_generator, optimizer, epoch, self.config)
                logger.info('-' * 89)

                val_loss, val_recall, val_precision, val_f1 = validate_model(self.model_instance, self.data_generator, epoch, self.config)
                logger.info('-' * 89)

                if not best_val_loss or val_loss < best_val_loss:
                    with open(checkpoint_dir.format(epoch=epoch, val_loss=val_loss), 'wb') as f:
                        torch.save(self.model_instance, f)
                    best_val_loss = val_loss
                    self.best_epoch = epoch
        except KeyboardInterrupt:
            logger.info('*' * 89)
            logger.info('Early Stopping!')

    def _serialize_model(self):
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "model"))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        logger.info('Save into {0}'.format(out_dir))
        model_name = out_dir + '/best_model.model'
        with open(model_name, 'wb') as f:
            torch.save(self.model_instance, f)
        # model_name = out_dir + '/model-{timestamp}.model'
        # with open(model_name.format(timestamp=str(int(time.time()))), 'wb') as f:
        #     torch.save(self.model_instance, f)


class DreamModelInference():
    def __init__(self, data_generator, **kwargs):
        self.config = Config.from_dict(kwargs)
        # self.batch_size = batch_size
        # self.config.num_product = data_generator.num_product
        # self.config.seq_len = data_generator.seq_len
        # self.kwargs = kwargs
        self.model_instance = None
        self.data_generator = data_generator
        # self.best_epoch = 0

    def _deserialize_model(self):
        model_dir = os.path.abspath(os.path.join(os.path.curdir, "model", "best_model.model"))
        self.model_instance = torch.load(model_dir)
        # deserialized_model = pickle.loads(b'\x80\x03K\x01.')
        # return model

    def _do_evaluation(self):
        # TODO remove the model initialization
        # self._do_evaluation()
        batch_predict(self.model_instance, self.data_generator, self.config)

    def _get_performance_metrics(self, **kwargs):
        # TODO: the validation_model need epoch which is ot necessary for evaluation
        # TODO: The validation_model print the output that is not necessary for this step
        # TODO remove the model initialization
        # self._do_evaluation()
        val_loss, val_recall, val_precision, val_f1 = validate_model(self.model_instance, self.data_generator, 0,
                                                                     self.config)

        return {"Recall": val_recall, "precision" : val_precision, "f1_score" : val_f1}
batch_size = 1
num_product = 10
seq_len = 4
#
data_generator = ReturnsDataGen(batch_size, num_product, seq_len)
# # print(data_generator._batch_size)
# recomendation = DreamModelTraining(data_generator, top_k = 10)
# recomendation._do_training()
# recomendation._serialize_model()

infrence = DreamModelInference(data_generator)
infrence._deserialize_model()
infrence._do_evaluation()
infrence._get_performance_metrics()