FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Preinstall Torch CPU + MLflow so jobs don’t download huge wheels each run
RUN python -m pip install --no-cache-dir \
      torch==2.4.0 && \
    python -m pip install --no-cache-dir mlflow==2.14.1

