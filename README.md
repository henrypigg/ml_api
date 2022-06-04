# ML Training and Prediction API
## Development
Deploy Kubernetes Cluster
```
minikube start
eval $(minikube -p minikube docker-env)
docker build --tag ml-backend:1 --file Dockerfile .
kubectl apply -f rabbitmq.yaml
kubectl apply -f api.yaml
kubectl apply -f workers.yaml
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
kubectl create namespace keda
kubectl apply -f keda.yaml
kubectl port-forward --namespace ml-backend-api service/fastapi-server 8080:80
```

## API Calls
```
/train
    POST
    /{task_id}
        GET
/predict
    POST
    /{task_id}
        GET
```
### `/train`
Request Model:
```
{
    "dataset_filename": str,
    "label_columns": List[str],
    "target_column": str,
    "model_type": str
}
```
Response Model:
```
{
    "task_id": str,
    "status": str,
    "model_info": {
        "model_id": str,
        "metric_key": str,
        "metric": str    
    }
}
```

### `/predict`
Request Model:
```
{
    "model_id": str,
    "data": List[float]
}
```
Response Model:
```
{
    task_id: str,
    status: str,
    prediction: str
}
```
