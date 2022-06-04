from pydantic import BaseModel
from typing import List, Dict


class Task(BaseModel):
    task_id: str
    status: str

class PredictRequest(BaseModel):
    model_id: str
    data: List[List[float]]

class PredictResponse(BaseModel):
    task_id: str
    status: str
    prediction: str

class TrainRequest(BaseModel):
    dataset_filename: str
    label_columns: List[str]
    target_column: str
    model_type: str

class TrainResponse(BaseModel):
    task_id: str
    status: str
    model_info: Dict[str, str]