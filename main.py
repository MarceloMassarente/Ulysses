import os
import time
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

# Set environment variables for CPU threading before PyTorch/transformers loads
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8")
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "8")

model_id = "Bilaal/ulysses-ner-br"
ner_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_pipeline
    print(f"Loading model {model_id}...")
    
    # 1. Configurando Threads do PyTorch dinamicamente
    # No Railway (Docker restrito), os.cpu_count() pode vazar os núcleos da máquina host.
    # sched_getaffinity() garante que leremos apenas a cota de CPU liberada para o container.
    try:
        available_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cores = os.cpu_count() or 2

    # Presume-se hyperthreading. Para focar em núcleos reais, dividimos por 2 (mínimo de 1).
    physical_cores = max(1, available_cores // 2)
    torch.set_num_threads(physical_cores)
    print(f"Ambiente detectou {available_cores} vCPUs. PyTorch threads setadas para {physical_cores}.")
    
    # 2. Carregando modelo e tokenizador base
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    
    # 3. Dynamic Quantization (Ouro para CPU-only)
    # Converte os pesos das camadas lineares para INT8 em tempo real.
    # Reduz uso de RAM pela metade e dobra a velocidade no Ryzen!
    print("Applying Dynamic Quantization (INT8) for CPU acceleration...")
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # 4. Criando o pipeline otimizado
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    print("Model loaded and optimized successfully.")
    yield
    print("Shutting down model...")
    ner_pipeline = None

app = FastAPI(
    title="Ulysses Legal NER API",
    description="Microservice for Legal Entity Extraction using UlyssesNER-BR",
    version="1.0.0",
    lifespan=lifespan
)

# Configuração de CORS para permitir acesso de qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens. Para produção, especifique os domínios.
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, etc)
    allow_headers=["*"],  # Permite todos os cabeçalhos
)

class ExtractRequest(BaseModel):
    text: str = Field(..., description="The raw text to be processed")
    confidence_threshold: float = Field(0.0, description="Minimum confidence score for extracted entities")

class Entity(BaseModel):
    entity_group: str
    word: str
    score: float

class ExtractResponse(BaseModel):
    status: str
    processing_time_ms: int
    entities: List[Entity]

@app.get("/health")
async def health_check():
    if ner_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok", "model": model_id}

@app.post("/api/v1/extract", response_model=ExtractResponse)
async def extract_entities(request: ExtractRequest):
    if ner_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Perform inference
        predictions = ner_pipeline(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    entities = []
    for pred in predictions:
        score = float(pred["score"])
        if score >= request.confidence_threshold:
            # Clean up words from subwords artifacts if necessary
            word = pred["word"].strip()
            # simple aggregation sometimes returns words with '##' or spaces. 
            # transformers handles it but we just make sure.
            
            entities.append(Entity(
                entity_group=pred["entity_group"],
                word=word,
                score=score
            ))
            
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    return ExtractResponse(
        status="success",
        processing_time_ms=processing_time_ms,
        entities=entities
    )
