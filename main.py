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

model_id = "dominguesm/legal-bert-ner-base-cased-ptbr"
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

def clean_entity_text(text: str) -> str:
    # 1. Remove spacing artifacts around dots, hyphens, slashes, and underscores
    # e.g., "8 . 666 / 1993" -> "8.666/1993", "1010516 - 78" -> "1010516-78"
    text = re.sub(r'\s*([.\-/_])\s*', r'\1', text)
    
    # 2. Fix spacing around commas (no space before, one space after)
    text = re.sub(r'\s*,\s*', ', ', text)
    
    # 3. Remove subword markers "##" if any got through
    text = text.replace("##", "")
    
    # 4. Remove leading/trailing punctuation, symbols, and spaces
    text = re.sub(r'^[.,\-/_;\s]+', '', text)
    text = re.sub(r'[.,\-/_;\s]+$', '', text)
    
    # 5. Collapse double spaces to single spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

import re

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
        # BERT has a strict 512-token limit. For long legal texts (like a 30-page PDF),
        # we must split the text into smaller overlapping chunks to avoid cutting entities.
        # PDF text extractors (like pypdf) often introduce hard line breaks '\n' inside sentences.
        # We replace single newlines with spaces to reconstruct sentences while preserving paragraph breaks.
        cleaned_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', request.text)
        cleaned_text = re.sub(r'\n+', '\n', cleaned_text)  # collapse multiple newlines
        
        # Split text into words (approximate tokens)
        words = cleaned_text.split()
        
        # Sliding Window Chunking
        chunk_size = 100  # Words per chunk (~130 BERT tokens, safe margin under 512)
        overlap = 20      # Word overlap to capture entities at chunk boundaries
        
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i : i + chunk_size]
            chunks.append(" ".join(chunk_words))
            if i + chunk_size >= len(words):
                break
            i += chunk_size - overlap

        # Perform inference sequentially on each chunk
        predictions = []
        for chunk in chunks:
            if not chunk:
                continue
            
            try:
                # Safeguard: tokenize and truncate to 512 tokens, then decode back to a string.
                # This handles table/URL token spikes or exceptionally dense paragraphs perfectly,
                # completely eliminating any possibility of tensor size mismatches across all transformers versions.
                tokenized = ner_pipeline.tokenizer(chunk, truncation=True, max_length=512)
                chunk = ner_pipeline.tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)
                
                chunk_preds = ner_pipeline(chunk)
                predictions.extend(chunk_preds)
            except Exception as e:
                print(f"Erro ao processar chunk: {e}")
                continue
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na inferência: {str(e)}")
    
    # Deduplicate and sanitize predicted entities
    entities_map = {}
    for pred in predictions:
        score = float(pred["score"])
        if score >= request.confidence_threshold:
            word = clean_entity_text(pred["word"])
            # Ignore empty or single-character entities
            if not word or len(word) <= 1:
                continue
            
            group = pred["entity_group"]
            key = (group, word)
            
            # Keep the highest confidence score for each unique entity
            if key not in entities_map or score > entities_map[key]:
                entities_map[key] = score
                
    entities = [
        Entity(entity_group=group, word=word, score=score)
        for (group, word), score in entities_map.items()
    ]
            
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    return ExtractResponse(
        status="success",
        processing_time_ms=processing_time_ms,
        entities=entities
    )
