FROM nvidia/cuda:12.1.1-runtime-ubi9
RUN microdnf install -y python3.11 python3.11-pip && microdnf clean all
RUN python3.11 -m pip install --no-cache-dir \
      torch==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121 \
      mlflow==2.14.1
WORKDIR /app