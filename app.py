from fastapi import FastAPI
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
from models import PredictRequest, PredictResponse, TrainRequest, TrainResponse, Task

from src.tasks import predict_single, train_model
from src.worker import celery

app = FastAPI()


@app.post("/predict", response_model=Task, status_code=202)
def predict(request: PredictRequest):
  task_id = predict_single.delay(request.data, request.model_id)
  return {'task_id': str(task_id), 'status': 'Processing'}

@app.get('/predict/{task_id}', response_model=PredictResponse, status_code=200)
def predict(task_id):
  task = celery.AsyncResult(task_id)
  if not task.ready():
    print(app.url_path_for('predict'))
    return JSONResponse(
      status_code=202, 
      content={
        'task_id': str(task_id),
        'status': 'Processing'
      }
    )
  result = task.get()
  return {'task_id': task_id, 'status': 'Success', 'prediction': str(result)}

@app.post("/train", response_model=Task, status_code=202)
def train(request: TrainRequest):
  task_id = train_model.delay(
    request.dataset_filename,
    request.label_columns,
    request.target_column,
    request.model_type
  )
  return {'task_id': str(task_id), 'status': 'Processing'}

@app.get("/train/{task_id}", response_model=TrainResponse, status_code=200)
def train(task_id):
  task = celery.AsyncResult(task_id)
  if not task.ready():
    print(app.url_path_for('train'))
    return JSONResponse(
      status_code=202,
      content={
        'task_id': str(task_id),
        'status': 'Processing'
      }
    )
  model_info = task.get()
  return {
    'task_id': str(task_id), 
    'status': 'Success',
    'model_info': model_info
  }