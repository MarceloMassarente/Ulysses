"""Ulysses Legal NER API — microserviço de extração de entidades jurídicas."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

logger = logging.getLogger("ulysses_ner")

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "8"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "8"))

MODEL_ID = os.environ.get(
    "LEGAL_NER_MODEL", "dominguesm/legal-bert-ner-base-cased-ptbr"
)
MAX_LENGTH = int(os.environ.get("LEGAL_NER_MAX_LENGTH", "512"))
STRIDE = int(os.environ.get("LEGAL_NER_STRIDE", "128"))
AGGREGATION = os.environ.get("LEGAL_NER_AGGREGATION", "first")

REGEX_PATTERNS: dict[str, str] = {
    "CPF": r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",
    "CNPJ": r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b",
    "PROCESSO": r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b",
    "LEGISLACAO": (
        r"\b(?:[Aa]rt(?:\.?o?|igo)?\.?\s+\d+[°º]?"
        r"|[Ll]ei\s+n?\.?\s*\d+(?:[.\d]+)*(?:/\d+)?)\b"
    ),
    "DATA": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    "VALOR": r"\bR\$\s*[\d.,]+\b",
}

ner_pipeline: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_pipeline
    logger.info("Loading model %s...", MODEL_ID)

    try:
        available_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cores = os.cpu_count() or 2

    physical_cores = max(1, available_cores // 2)
    torch.set_num_threads(physical_cores)
    logger.info(
        "Detected %s vCPUs; PyTorch threads=%s", available_cores, physical_cores
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

    if not tokenizer.is_fast and AGGREGATION != "simple":
        logger.warning(
            "Slow tokenizer: falling back to aggregation_strategy=simple"
        )
        agg = "simple"
    else:
        agg = AGGREGATION

    logger.info("Applying dynamic quantization (INT8)...")
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy=agg,
    )
    logger.info("Model loaded (aggregation=%s).", agg)
    yield
    logger.info("Shutting down model...")
    ner_pipeline = None


app = FastAPI(
    title="Ulysses Legal NER API",
    description="Microservice for Legal Entity Extraction using legal-bert-ner",
    version="1.1.0",
    lifespan=lifespan,
)

_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    text: str = Field(..., description="The raw text to be processed")
    confidence_threshold: float = Field(
        0.0, description="Minimum confidence score for extracted entities"
    )
    include_regex: bool = Field(
        False,
        description="Merge high-precision regex entities (CPF, CNPJ, processo, etc.)",
    )


class Entity(BaseModel):
    entity_group: str
    text: str
    word: str
    score: float
    start: int | None = None
    end: int | None = None
    source: str = "ner"


class ExtractResponse(BaseModel):
    status: str
    processing_time_ms: int
    entities: list[Entity]


def clean_entity_text(text: str) -> str:
    text = re.sub(r"\s*([.\-/_])\s*", r"\1", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = text.replace("##", "")
    text = re.sub(r"^[.,\-/_;\s]+", "", text)
    text = re.sub(r"[.,\-/_;\s]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_pdf_text(text: str) -> str:
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return re.sub(r"\n+", "\n", text)


def is_noisy_entity(group: str, word: str) -> bool:
    if not word or len(word) <= 1:
        return True
    if group == "TEMPO" and (
        len(word) <= 3 or re.fullmatch(r"[\d\s./-]+", word) is not None
    ):
        return True
    if len(word) <= 2 and word.isupper() and group in ("PESSOA", "ORGANIZACAO", "LOCAL"):
        return True
    return False


def regex_entities(text: str, threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, pattern in REGEX_PATTERNS.items():
        for match in re.finditer(pattern, text):
            word = clean_entity_text(match.group(0))
            if not word or is_noisy_entity(label, word):
                continue
            out.append(
                {
                    "entity_group": label,
                    "word": word,
                    "score": 0.99,
                    "start": match.start(),
                    "end": match.end(),
                    "source": "regex",
                }
            )
    _ = threshold
    return out


def run_ner_inference(cleaned_text: str) -> list[dict[str, Any]]:
    if not cleaned_text.strip():
        return []

    pipe = ner_pipeline
    kwargs: dict[str, Any] = {
        "truncation": True,
        "max_length": MAX_LENGTH,
    }
    if pipe.tokenizer.is_fast and len(cleaned_text) > 400:
        kwargs["stride"] = STRIDE

    raw = pipe(cleaned_text, **kwargs)
    if isinstance(raw, dict):
        return [raw]
    return list(raw)


def merge_predictions(
    predictions: list[dict[str, Any]],
    threshold: float,
) -> list[Entity]:
    # Dedup by (label, span text); stride overlap may repeat the same entity with other offsets.
    entities_map: dict[tuple[str, str], dict[str, Any]] = {}

    for pred in predictions:
        score = float(pred.get("score", 0))
        if score < threshold:
            continue

        group = str(pred.get("entity_group", ""))
        word = clean_entity_text(str(pred.get("word", "")))
        if is_noisy_entity(group, word):
            continue

        start = pred.get("start")
        end = pred.get("end")
        start_i = int(start) if isinstance(start, int) else None
        end_i = int(end) if isinstance(end, int) else None

        key = (group, word)
        existing = entities_map.get(key)
        if existing is None or score > float(existing["score"]):
            entities_map[key] = {
                "entity_group": group,
                "word": word,
                "score": score,
                "start": start_i,
                "end": end_i,
                "source": pred.get("source", "ner"),
            }

    entities = [
        Entity(
            entity_group=item["entity_group"],
            text=item["word"],
            word=item["word"],
            score=item["score"],
            start=item["start"],
            end=item["end"],
            source=item["source"],
        )
        for item in entities_map.values()
    ]
    entities.sort(key=lambda e: (-e.score, e.entity_group, e.text))
    return entities


@app.get("/health")
async def health_check():
    if ner_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {
        "status": "ok",
        "model": MODEL_ID,
        "aggregation": AGGREGATION,
        "max_length": MAX_LENGTH,
        "stride": STRIDE,
    }


@app.post("/api/v1/extract", response_model=ExtractResponse)
async def extract_entities(request: ExtractRequest):
    if ner_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        cleaned_text = normalize_pdf_text(request.text)
        predictions = await asyncio.to_thread(run_ner_inference, cleaned_text)

        if request.include_regex:
            predictions.extend(regex_entities(cleaned_text, request.confidence_threshold))

        entities = merge_predictions(predictions, request.confidence_threshold)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Erro na inferência: {e}") from e

    processing_time_ms = int((time.time() - start_time) * 1000)
    return ExtractResponse(
        status="success",
        processing_time_ms=processing_time_ms,
        entities=entities,
    )
