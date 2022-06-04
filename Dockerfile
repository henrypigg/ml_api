FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt /app/

RUN python3.8 -m pip install --upgrade pip && \
    python3.8 -m pip install -U setuptools && \
    python3.8 -m pip install -r requirements.txt --no-cache-dir

RUN python3.8 -m pip install s3fs

COPY . /app/

CMD uvicorn app:app --host 0.0.0.0 --port 8080
