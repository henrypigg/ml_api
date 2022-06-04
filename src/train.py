from cProfile import label
import json
import boto3
import os
import pickle
import pandas as pd
import uuid
import shutil

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SESSION = boto3.Session(
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
)
S3 = SESSION.client('s3', region_name='us-west-1')
S3_BUCKET = os.environ['S3_BUCKET']
MODEL_PATH = os.environ['MODEL_PATH']


def _train_decision_tree(x, y):
    return DecisionTreeClassifier().fit(x.values, y.values)


def _train_sgd_classifier(x, y):
    return SGDClassifier().fit(x.values, y.values)


def _train_k_nearest_neighbors(x, y):
    return KNeighborsClassifier().fit(x.values, y.values)


class ModelTrainer:
    """
    Wrapper for the model training and storing
    """
    def __init__(self):
        self.df = None
        self.label_columns = []

        self.target_column = None
        self.model_type = None
        self.metric_key = None
        self.model_id = str(uuid.uuid4())

    def setup(self, dataset_filename, label_columns, target_column, model_type):
        self.model_type = model_type
        if dataset_filename == 'TEST_DATASET':
            self.df = pd.DataFrame(
                columns=['a', 'b', 'c'],
                data=[
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [0, 0, 0]
                ]
            )
            self.label_columns = ['a', 'b']
            self.target_column = 'c'
        else:
            self.label_columns = label_columns
            self.target_column = target_column
            self.df = self._load_dataset(dataset_filename)
    
    def train(self, test_size=0.15):
        train_data, test_data = train_test_split(
            self.df,
            test_size=test_size,
            random_state=42
        )

        model = self._train(train_data)

        if self.model_type == 'AutoML':
            self.metric_key, metric = self.evaluate_automl_model(model, test_data)
        else:
            y_pred = model.predict(test_data[self.label_columns])
            metric = accuracy_score(y_pred, test_data[self.target_column])
            self.metric_key = 'accuracy'

        self._store_model(model, metric)

        return {
            'model_id': self.model_id,
            'metric_type': self.metric_key,
            'metric': metric
        }

    def evaluate_automl_model(self, model, test_data):
        model_evaluation = model.evaluate_predictions(
            y_true=test_data[self.target_column],
            y_pred=model.predict(test_data.drop(self.target_column, axis=1)),
            auxiliary_metrics=True
        )
        metric_key = 'accuracy' if 'accuracy' in model_evaluation else 'r2'
        return metric_key, model_evaluation.get(metric_key)
    
    def _load_dataset(self, dataset_filename, size=None):
        if not os.path.exists('/tmp/'):
            os.mkir('/tmp/')

        S3.download_file(
            S3_BUCKET,
            os.path.join('datasets', dataset_filename),
            '/tmp/dataset.csv'
        )

        df = pd.read_csv('/tmp/dataset.csv')
        df = df.dropna(subset=self.label_columns + [self.target_column])
        df = df.sample(frac=1)
        if size and len(df) > size:
            df = df[:size]

        return df

    def _store_model(self, model, metric):
        # Store model info
        S3.put_object(
            Bucket=S3_BUCKET,
            Body=json.dumps({
                'model_type': self.model_type,
                'target_column': self.target_column,
                'label_columns': self.label_columns,
                'metric_key': self.metric_key,
                'metric': metric
            }),
            Key=os.path.join(MODEL_PATH, self.model_id, 'model_info.json')
        )

        # Store model
        S3.put_object(
            Bucket=S3_BUCKET,
            Body=pickle.dumps(model),
            Key=os.path.join(MODEL_PATH, self.model_id, 'model.pkl')
        )

    def _store_autogluon_models(self):
        shutil.make_archive('/tmp/AutogluonModels', 'zip', base_dir='AutogluonModels', root_dir='.')

        S3.upload_file(
            Filename='/tmp/AutogluonModels.zip',
            Bucket=S3_BUCKET,
            Key=os.path.join(MODEL_PATH, self.model_id, 'AutogluonModels.zip')
        )

    def _train(self, train_data):
        x, y = train_data[self.label_columns], train_data[self.target_column]
        if self.model_type == 'Decision Tree':
            model = _train_decision_tree(x, y)
        elif self.model_type == 'Linear Regression':
            model = _train_sgd_classifier(x, y)
        elif self.model_type == 'K-Nearest Neighbors':
            model = _train_k_nearest_neighbors(x, y)
        elif self.model_type == 'AutoML':
            model = self._train_auto_ml(train_data[self.label_columns + [self.target_column]])
        else:
            raise ValueError('Model type not supported')

        return model

    def _train_auto_ml(self, train_data):
        k = 500
        predictor = TabularPredictor(label=self.target_column).fit(train_data.head(k), hyperparameters={'GBM':{}, 'CAT':{}, 'RF':{}, 'XT':{}, 'KNN':{}})
        self._store_autogluon_models()
        return predictor
        
