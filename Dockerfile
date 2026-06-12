FROM python:3.11-slim

WORKDIR /app

ARG LEGAL_NER_MODEL=dominguesm/legal-bert-ner-base-cased-ptbr
ARG LEGAL_NER_MODEL_REVISION=4421092
ENV LEGAL_NER_MODEL=${LEGAL_NER_MODEL}
ENV LEGAL_NER_MODEL_REVISION=${LEGAL_NER_MODEL_REVISION}

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ libjemalloc2 && rm -rf /var/lib/apt/lists/*

# Optimize memory allocation for multi-threaded CPU workloads
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

# Install PyTorch CPU directly first to avoid pulling the massive CUDA version during requirements install
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model into the image during build time
RUN python -c "import os; from transformers import pipeline; pipeline('ner', model=os.environ['LEGAL_NER_MODEL'], revision=os.environ['LEGAL_NER_MODEL_REVISION'], aggregation_strategy='simple')"

COPY . /app

EXPOSE 5522

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-5522} --workers 1
