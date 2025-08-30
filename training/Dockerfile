FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch (CPU build) + related packages
RUN pip install torch==2.4.0+cpu torchvision==0.19.0+cpu torchaudio==2.4.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html
