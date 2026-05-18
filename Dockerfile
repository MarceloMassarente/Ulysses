FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ libjemalloc2 && rm -rf /var/lib/apt/lists/*

# Optimize memory allocation for multi-threaded CPU workloads
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

# Install PyTorch CPU directly first to avoid pulling the massive CUDA version during requirements install
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Pre-download the model into the image during build time
RUN python -c "from transformers import pipeline; pipeline('ner', model='dominguesm/legal-bert-ner-base-cased-ptbr', aggregation_strategy='simple')"

EXPOSE 8000

# Uso do shell syntax (sem colchetes) para avaliar variáveis de ambiente dinamicamente.
# No Railway, a variável $PORT é definida automaticamente pelo orquestrador deles.
# O fallback `:-8000` garante que localmente, via docker-compose, ele use a porta 8000.
# Também tornamos o número de workers flexível (default 1 para Railway, evitando OOM em planos gratuitos).
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1}
