import logging
from celery import Task
from src.predict import Predictor
from src.train import ModelTrainer

from .worker import celery

class PredictTask(Task):
    """
    Abstraction of Celery's Task class to support prediction
    """
    abstract = True

    def __init__(self):
        super().__init__()
        self.predictor = None
    
    def __call__(self, *args, **kwargs):
        """
        Load predictor on first call
        """
        if not self.predictor:
            logging.info('Loading Predictor...')
            self.predictor = Predictor()
            logging.info('Predictor Loaded')
        return self.run(*args, **kwargs)

class TrainTask(Task):
    """
    Abstraction of Celery's Task class to support training
    """
    abstract = True

    def __init__(self):
        super().__init__()
        self.model_trainer = None

    def __call__(self, *args, **kwargs):
        """
        Load model trainer on first call
        """
        if not self.model_trainer:
            logging.info('Loading Model Trainer...')
            self.model_trainer = ModelTrainer()
            logging.info('Model Trainer Loaded')
        return self.run(*args, **kwargs)


@celery.task(
    ignore_result=False,
    bind=True,
    base=PredictTask,
    name=f'{__name__}.Predict'
)
def predict_single(self, data, model_id):
    """
    Run method of PredictTask
    """
    self.predictor.load_model(model_id)
    prediction = self.predictor.predict(data)[0]
    return str(prediction)


@celery.task(
    ignore_result=False,
    bind=True,
    base=TrainTask,
    name=f'{__name__}.Train'
)
def train_model(self, dataset_filename, label_columns, target_column, model_type):
    self.model_trainer.setup(
        dataset_filename,
        label_columns,
        target_column,
        model_type
    )
    model_info = self.model_trainer.train()
    return model_info


