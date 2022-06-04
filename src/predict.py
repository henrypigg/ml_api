import json
import shutil
import boto3
import os
import pickle
import pandas as pd

from typing import List

S3 = boto3.client('s3', region_name='us-west-1')
S3_BUCKET = os.environ['S3_BUCKET']
MODEL_PATH = os.environ['MODEL_PATH']

class Predictor:
    """
    Wrapper for loading and predicting against pre-trained model
    """
    def __init__(self):
        self.model = None

    def _load_autogluon_models(self, model_id: str):
        S3.download_file(
            S3_BUCKET,
            os.path.join(MODEL_PATH, model_id, 'AutogluonModels.zip'), 
            '/tmp/AutogluonModels.zip'
        )

        shutil.unpack_archive('/tmp/AutogluonModels.zip', '/app/', 'zip')

    def _load_model(self, model_name: str):
        if not os.path.exists('/tmp/'):
            os.mkdir('/tmp/')

        S3.download_file(
            S3_BUCKET,
            os.path.join(MODEL_PATH, model_name),
            '/tmp/model.pkl'
        )

        with open('/tmp/model.pkl', 'rb') as f:
            self.model = pickle.load(f)
    
    def _load_model_info(self, model_id):
        S3.download_file(
            S3_BUCKET,
            os.path.join(MODEL_PATH, model_id, 'model_info.json'),
            '/tmp/model_info.json'
        )

        with open('/tmp/model_info.json', 'rb') as f:
            self.model_info = json.load(f) 

    def load_model(self, model_id):    
        self._load_model_info(model_id)
        if self.model_info.get('model_type') == 'AutoML':
            self._load_autogluon_models(model_id)
        self._load_model(os.path.join(model_id, 'model.pkl'))

    def predict(self, data: List[float]) -> float:
        if not self.model:
            raise ValueError("Model required for prediction. No model loaded.")
        elif self.model_info.get("model_type") == "AutoML":
            return self.model.predict(
                pd.DataFrame(data, columns=self.model_info["label_columns"])
            )
        return self.model.predict(data)

    def predict_list(self, data_list: List[List[float]]) -> List[float]:
        if not self.model:
            raise ValueError("Model required for prediction. No model loaded.")
        return [self.model.predict(data) for data in data_list]