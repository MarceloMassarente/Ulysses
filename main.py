"""Ulysses Legal NER API — microserviço de extração de entidades jurídicas."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import threading
import time
import unicodedata
from contextlib import asynccontextmanager
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "8"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "8"))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

logger = logging.getLogger("ulysses_ner")

MODEL_ID = os.environ.get(
    "LEGAL_NER_MODEL", "dominguesm/legal-bert-ner-base-cased-ptbr"
)
MODEL_REVISION = os.environ.get("LEGAL_NER_MODEL_REVISION", "4421092")
MAX_LENGTH = int(os.environ.get("LEGAL_NER_MAX_LENGTH", "512"))
STRIDE = int(os.environ.get("LEGAL_NER_STRIDE", "128"))
MAX_INPUT_CHARS = int(os.environ.get("LEGAL_NER_MAX_INPUT_CHARS", "50000"))
DIRECT_PIPE_MAX_CHARS = int(os.environ.get("LEGAL_NER_DIRECT_PIPE_MAX_CHARS", "5000"))
MAX_IN_FLIGHT = max(1, int(os.environ.get("LEGAL_NER_MAX_IN_FLIGHT", "4")))
QUEUE_TIMEOUT_SECONDS = max(
    0.0, float(os.environ.get("LEGAL_NER_QUEUE_TIMEOUT_SECONDS", "5"))
)
REQUEST_TIMEOUT_SECONDS = max(
    0.0, float(os.environ.get("LEGAL_NER_REQUEST_TIMEOUT_SECONDS", "0"))
)
AGGREGATION = os.environ.get("LEGAL_NER_AGGREGATION", "first")
JURISPRUDENCIA_NER_THRESHOLD = float(
    os.environ.get("LEGAL_NER_JURISPRUDENCIA_THRESHOLD", "0.35")
)
EXPOSE_HEALTH_CONFIG = os.environ.get("EXPOSE_HEALTH_CONFIG", "false").lower() in (
    "1",
    "true",
    "yes",
)
ENABLE_API_DOCS = os.environ.get("ENABLE_API_DOCS", "false").lower() in (
    "1",
    "true",
    "yes",
)

REGEX_PATTERNS: dict[str, str] = {
    "CPF": r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
    "CNPJ": r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b",
    "PROCESSO": r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b",
    "LEGISLACAO": (
        r"\b(?:§\s*\d+[°º]?\s+(?:do\s+)?)?"
        r"(?:[Aa]rt(?:\.?o?|igo)?\.?\s+\d+[°º]?"
        r"|[Ll]ei\s+n?[°º.]?\.?\s*\d+(?:[.\d]+)*(?:/\d+)?)\b"
    ),
    "DATA": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    "VALOR": (
        r"\b(?:R\$\s*[\d.,]+|\d{1,3}(?:\.\d{3})*,\d{2}\s+reais)\b"
    ),
}

# Jurisprudência: alinhado ao citation extractor do RAGjuridico (verification/extractor.py)
_RE_SUMULA = re.compile(
    r"\bS[uú]mula\s+"
    r"(?:Vinculante\s+)?"
    r"(?:n\.?\s*)?"
    r"\d+"
    r"(?:\s*(?:do|da|de|/)\s*(?:STJ|STF|TST|TSE|STM))?"
    r"\b",
    re.IGNORECASE,
)
_JURIS_MARKERS = re.compile(
    r"\b(?:S[uú]mula|REsp|AREsp|AgInt|AgRg|EREsp|HC|RHC|MS|RMS|ADI|ADC|ADPF|"
    r"ARE|RE|AI|ACO|MI|PET|RCL|STJ|STF|TST|TSE|STM|Acórdão|Acordao)\b",
    re.IGNORECASE,
)
_RE_ACORDAO = re.compile(
    r"\b(REsp|AREsp|AgInt|AgRg|EREsp|HC|RHC|MS|RMS|ADI|ADC|ADPF|ARE|RE|AI|ACO|MI|PET|RCL)"
    r"\.?\s*(?:n\.?\s*)?(\d[\d.]*(?:\/[A-Z]{2})?)",
    re.IGNORECASE,
)
_RE_ACORDAO_NOME = re.compile(
    r"\b[Aa]c[oó]rd[aã]o\s+(?:n\.?\s*)?(\d[\d./-]*)",
    re.IGNORECASE,
)
_RE_PROCESSO_CNJ = re.compile(r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b")
_RE_TEMA = re.compile(
    r"\bTema\s+(?:n[°º.]?\s*)?\d+"
    r"(?:\s*(?:do|da|de)\s+(?:STF|STJ|TST|TSE|STM))?\b",
    re.IGNORECASE,
)
_RE_RECURSO_ESTADUAL = re.compile(
    r"\b(?:Apelação|Apelacao|Agravo|Embargos|Recurso|Reexame)"
    r"\s+(?:Cível|Civel|Inominado|Ordinário|Ordinario|Extraordinário|Extraordinario)?"
    r"(?:\s+n[°º.]?\.?\s*\d[\d./-]*|\s+\d{7,}[\d./-]*)",
    re.IGNORECASE,
)
_RE_ADVOGADO = re.compile(
    r"\b(?:Dr\.?a?|Dra\.?)\s+"
    r"([A-ZÁÉÍÓÚÂÊÔÃÇ][a-záéíóúâêôãç]+"
    r"(?:\s+(?:da|de|do|dos|das)\s+[A-ZÁÉÍÓÚÂÊÔÃÇ][a-záéíóúâêôãç]+)*)",
    re.UNICODE,
)
_RE_COMARCA = re.compile(
    r"\bComarca\s+de\s+([A-ZÁÉÍÓÚÂÊÔÃÇ][a-záéíóúâêôãç]+(?:\s+[A-Za-záéíóúâêôãç.]+)*)",
    re.IGNORECASE,
)
_RE_FORO = re.compile(
    r"\bForo\s+([A-ZÁÉÍÓÚÂÊÔÃÇ][^\n,]{3,50})",
    re.IGNORECASE,
)
_RE_DATA_EXTENSO = re.compile(
    r"\b(\d{1,2})\s+de\s+"
    r"(janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|"
    r"setembro|outubro|novembro|dezembro)\s+de\s+(\d{4})\b",
    re.IGNORECASE,
)
_RE_DOUTRINA_CITACAO = re.compile(
    r"\b([A-ZÁÉÍÓÚÂÊÔÃÇ][\wáéíóúâêôãç-]+),\s*\d{4},\s*p\.?\s*\d+\b",
    re.UNICODE,
)
_RE_DOUTRINA_DE = re.compile(
    r"\bdoutrina\s+de\s+([A-ZÁÉÍÓÚÂÊÔÃÇ][^\n,.]{3,80})",
    re.IGNORECASE,
)
_RE_ORG_SA = re.compile(
    r"\b([A-ZÁÉÍÓÚÂÊÔÃÇ][\wáéíóúâêôãç.]*(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÇ][\wáéíóúâêôãç.]*)*)\s+S\.?A\.?\b",
    re.UNICODE,
)

_MESES: dict[str, int] = {
    "janeiro": 1,
    "fevereiro": 2,
    "março": 3,
    "marco": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}

REGEX_JURISPRUDENCIA: tuple[re.Pattern[str], ...] = (
    _RE_SUMULA,
    _RE_ACORDAO,
    _RE_ACORDAO_NOME,
    _RE_TEMA,
    _RE_RECURSO_ESTADUAL,
)

_NOISY_ORG_FRAGMENTS = frozenset(
    {"CO", "NI", "ST", "DF", "RC", "MA", "RI", "ER", "IR", "EL", "AN", "CE", "VA", "SP"}
)
_UPPERCASE_CONNECTORS = frozenset({"DA", "DE", "DO", "DAS", "DOS", "E"})

ner_pipeline: Any = None
_infer_lock = threading.Lock()
_infer_slots = asyncio.Semaphore(MAX_IN_FLIGHT)


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

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True
    )
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )

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
    version="1.3.0",
    lifespan=lifespan,
    docs_url="/docs" if ENABLE_API_DOCS else None,
    redoc_url="/redoc" if ENABLE_API_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_API_DOCS else None,
)

_cors_origins = os.environ.get("CORS_ORIGINS", "").split(",")
_allowed_cors_origins = [o.strip() for o in _cors_origins if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_cors_origins,
    allow_credentials=bool(_allowed_cors_origins) and "*" not in _allowed_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)


class ExtractRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_INPUT_CHARS,
        description="The raw text to be processed",
    )
    confidence_threshold: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Minimum confidence score for extracted entities",
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


def normalize_legal_text(text: str) -> str:
    """Corrige typos frequentes de PDF/OCR antes do NER."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("Sumula", "Súmula").replace("sumula", "súmula")
    text = text.replace("Acordao", "Acórdão").replace("acordao", "acórdão")
    text = re.sub(r"\bC\?digo\b", "Código", text, flags=re.IGNORECASE)
    text = re.sub(r"\bC\?vel\b", "Cível", text, flags=re.IGNORECASE)
    text = re.sub(r"\bS\?o\b", "São", text, flags=re.IGNORECASE)
    return text


def normalize_pdf_text(text: str) -> str:
    text = normalize_legal_text(text)
    # Hifenização / quebra de linha em PDF (ex.: BAN-\nCO, BAN\nCO)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(
        r"\b([A-ZÁÉÍÓÚÂÊÔÃÇ]{2,})\s*\n\s*([A-ZÁÉÍÓÚÂÊÔÃÇ]{2,})\b",
        _join_uppercase_line_break,
        text,
    )
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return re.sub(r"\n+", "\n", text)


def _join_uppercase_line_break(match: re.Match[str]) -> str:
    left, right = match.group(1), match.group(2)
    if right in _UPPERCASE_CONNECTORS:
        return f"{left} {right}"
    if min(len(left), len(right)) <= 3:
        return f"{left}{right}"
    return f"{left} {right}"


def looks_like_process_number(text: str) -> bool:
    return _RE_PROCESSO_CNJ.search(text) is not None


def is_noisy_jurisprudencia_ner(word: str) -> bool:
    if len(word) < 5:
        return True
    if re.fullmatch(r"\d{4}", word):
        return True
    if not _JURIS_MARKERS.search(word):
        return True
    if looks_like_process_number(word):
        return True
    return False


def is_noisy_entity(group: str, word: str, source: str = "ner") -> bool:
    if not word or len(word) <= 1:
        return True
    if group == "PESSOA" and (
        word.upper() in ("OAB", "DR", "DRA", "PATRONO")
        or re.search(r"\bOAB\s*/", word, re.IGNORECASE) is not None
    ):
        return True
    if group == "TEMPO" and (
        len(word) <= 3 or re.fullmatch(r"[\d\s./-]+", word) is not None
    ):
        return True
    if (
        len(word) <= 3
        and word.isupper()
        and group in ("PESSOA", "ORGANIZACAO", "LOCAL")
        and word in _NOISY_ORG_FRAGMENTS
    ):
        return True
    if group == "JURISPRUDENCIA":
        if source == "regex":
            # Só descarta CNJ isolado; recursos formais (Apelação nº …) ficam.
            return _RE_PROCESSO_CNJ.fullmatch(word.strip()) is not None
        return is_noisy_jurisprudencia_ner(word)
    return False


def jurisprudence_regex_entities(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for pattern in REGEX_JURISPRUDENCIA:
        for match in pattern.finditer(text):
            word = clean_entity_text(match.group(0))
            if not word or is_noisy_entity("JURISPRUDENCIA", word, source="regex"):
                continue
            key = (word, match.start(), match.end())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "entity_group": "JURISPRUDENCIA",
                    "word": word,
                    "score": 0.99,
                    "start": match.start(),
                    "end": match.end(),
                    "source": "regex",
                }
            )
    return out


def reclassify_ner_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Súmula/REsp marcados como LEGISLACAO pelo modelo → JURISPRUDENCIA."""
    out: list[dict[str, Any]] = []
    for pred in predictions:
        if pred.get("source") == "regex":
            out.append(pred)
            continue
        word = clean_entity_text(str(pred.get("word", "")))
        group = str(pred.get("entity_group", ""))
        if group == "LEGISLACAO" and word:
            if _RE_SUMULA.search(word) or _RE_ACORDAO.search(word):
                pred = {**pred, "entity_group": "JURISPRUDENCIA"}
        out.append(pred)
    return out


def _append_regex_entity(
    out: list[dict[str, Any]],
    seen: set[tuple[str, int, int]],
    *,
    group: str,
    word: str,
    start: int,
    end: int,
) -> None:
    word = clean_entity_text(word)
    if not word or is_noisy_entity(group, word, source="regex"):
        return
    key = (group, start, end)
    if key in seen:
        return
    seen.add(key)
    out.append(
        {
            "entity_group": group,
            "word": word,
            "score": 0.99,
            "start": start,
            "end": end,
            "source": "regex",
        }
    )


def data_extenso_entities(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for match in _RE_DATA_EXTENSO.finditer(text):
        day_s, month_name, year_s = match.groups()
        month_num = _MESES.get(month_name.lower())
        if not month_num:
            continue
        word = f"{int(day_s):02d}/{month_num:02d}/{year_s}"
        _append_regex_entity(
            out,
            seen,
            group="DATA",
            word=word,
            start=match.start(),
            end=match.end(),
        )
    return out


def structured_regex_entities(text: str) -> list[dict[str, Any]]:
    """PESSOA, LOCAL, DOUTRINA e ORGANIZACAO via regex de alta precisão."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()

    for match in _RE_ADVOGADO.finditer(text):
        _append_regex_entity(
            out,
            seen,
            group="PESSOA",
            word=match.group(1),
            start=match.start(1),
            end=match.end(1),
        )

    for pattern, group in ((_RE_COMARCA, "LOCAL"), (_RE_FORO, "LOCAL")):
        for match in pattern.finditer(text):
            _append_regex_entity(
                out,
                seen,
                group=group,
                word=match.group(1),
                start=match.start(1),
                end=match.end(1),
            )

    for pattern in (_RE_DOUTRINA_CITACAO, _RE_DOUTRINA_DE):
        for match in pattern.finditer(text):
            word = match.group(1) if match.lastindex else match.group(0)
            _append_regex_entity(
                out,
                seen,
                group="DOUTRINA",
                word=word,
                start=match.start(1) if match.lastindex else match.start(),
                end=match.end(1) if match.lastindex else match.end(),
            )

    for match in _RE_ORG_SA.finditer(text):
        _append_regex_entity(
            out,
            seen,
            group="ORGANIZACAO",
            word=match.group(0),
            start=match.start(),
            end=match.end(),
        )

    return out


def regex_entities(text: str, threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for label, pattern in REGEX_PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            _append_regex_entity(
                out,
                seen,
                group=label,
                word=match.group(0),
                start=match.start(),
                end=match.end(),
            )
    out.extend(jurisprudence_regex_entities(text))
    out.extend(data_extenso_entities(text))
    out.extend(structured_regex_entities(text))
    _ = threshold
    return out


def _text_chunks(text: str, tokenizer: Any) -> list[tuple[str, int]]:
    """Split long text without passing tokenizer kwargs into pipeline.__call__."""
    if tokenizer.is_fast:
        with _infer_lock:
            encoded = tokenizer(
                text,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                max_length=MAX_LENGTH,
                stride=STRIDE,
                truncation=True,
            )
        chunks: list[tuple[str, int]] = []
        for offsets in encoded["offset_mapping"]:
            valid_offsets = [(start, end) for start, end in offsets if end > start]
            if not valid_offsets:
                continue
            start = min(start for start, _end in valid_offsets)
            end = max(end for _start, end in valid_offsets)
            chunk = text[start:end]
            if chunk:
                chunks.append((chunk, start))
        return chunks or [(text, 0)]

    words = text.split()
    chunk_words, overlap = 100, 20
    chunks: list[tuple[str, int]] = []
    i = 0
    search_from = 0
    while i < len(words):
        piece = " ".join(words[i : i + chunk_words])
        if piece:
            start = text.find(piece, search_from)
            if start < 0:
                start = search_from
            chunks.append((piece, start))
            search_from = max(start + 1, search_from)
        if i + chunk_words >= len(words):
            break
        i += chunk_words - overlap
    return chunks or [(text, 0)]


def _shift_prediction_offsets(
    predictions: list[dict[str, Any]], base_offset: int
) -> list[dict[str, Any]]:
    if base_offset == 0:
        return predictions
    shifted: list[dict[str, Any]] = []
    for pred in predictions:
        item = dict(pred)
        if isinstance(item.get("start"), int):
            item["start"] += base_offset
        if isinstance(item.get("end"), int):
            item["end"] += base_offset
        shifted.append(item)
    return shifted


def _run_pipe_on_text(pipe: Any, text: str) -> list[dict[str, Any]]:
    # transformers 5.x: only stride/aggregation_strategy accepted on __call__, not truncation.
    with _infer_lock:
        if pipe.tokenizer.is_fast and len(text) > 400 and STRIDE > 0:
            raw = pipe(text, stride=STRIDE)
        else:
            raw = pipe(text)
    if isinstance(raw, dict):
        return [raw]
    return list(raw)


def run_ner_inference(cleaned_text: str) -> list[dict[str, Any]]:
    if not cleaned_text.strip():
        return []

    pipe = ner_pipeline
    text = cleaned_text

    # Short text: single pipeline call (no stride).
    if len(text) <= 400:
        return _run_pipe_on_text(pipe, text)

    # Long text: stride when supported, else explicit token/word chunks.
    if pipe.tokenizer.is_fast and STRIDE > 0 and len(text) <= DIRECT_PIPE_MAX_CHARS:
        try:
            return _run_pipe_on_text(pipe, text)
        except (TypeError, ValueError) as exc:
            logger.warning("Stride inference failed (%s); falling back to chunks", exc)

    predictions: list[dict[str, Any]] = []
    for chunk, base_offset in _text_chunks(text, pipe.tokenizer):
        predictions.extend(
            _shift_prediction_offsets(_run_pipe_on_text(pipe, chunk), base_offset)
        )
    return predictions


def _effective_threshold(group: str, source: str, threshold: float) -> float:
    if source == "regex":
        return 0.0
    if group == "JURISPRUDENCIA":
        return min(threshold, JURISPRUDENCIA_NER_THRESHOLD)
    return threshold


def merge_predictions(
    predictions: list[dict[str, Any]],
    threshold: float,
) -> list[Entity]:
    # Dedup exact repeated spans from stride overlap while preserving repeated mentions.
    entities_map: dict[tuple[str, str, int | None, int | None], dict[str, Any]] = {}

    for pred in predictions:
        group = str(pred.get("entity_group", ""))
        source = str(pred.get("source", "ner"))
        score = float(pred.get("score", 0))
        if score < _effective_threshold(group, source, threshold):
            continue
        word = clean_entity_text(str(pred.get("word", "")))
        if is_noisy_entity(group, word, source=source):
            continue

        start = pred.get("start")
        end = pred.get("end")
        start_i = int(start) if isinstance(start, int) else None
        end_i = int(end) if isinstance(end, int) else None

        key = (group, word, start_i, end_i)
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


def process_extraction(
    text: str,
    confidence_threshold: float,
    include_regex: bool,
) -> list[Entity]:
    if not math.isfinite(confidence_threshold):
        raise ValueError("confidence_threshold must be finite")

    cleaned_text = normalize_pdf_text(text)
    predictions = run_ner_inference(cleaned_text)
    predictions = reclassify_ner_predictions(predictions)

    if include_regex:
        predictions.extend(regex_entities(cleaned_text, confidence_threshold))

    return merge_predictions(predictions, confidence_threshold)


async def _acquire_infer_slot() -> None:
    try:
        await asyncio.wait_for(_infer_slots.acquire(), timeout=QUEUE_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="NER inference is busy; retry later",
            headers={"Retry-After": str(max(1, int(QUEUE_TIMEOUT_SECONDS)))},
        ) from exc


def _release_slot_after_timeout(task: asyncio.Task[list[Entity]]) -> None:
    try:
        task.result()
    except Exception:
        logger.exception("Timed-out inference finished with error")
    finally:
        _infer_slots.release()


async def _run_extraction_with_slot(request: ExtractRequest) -> list[Entity]:
    await _acquire_infer_slot()
    worker_task = asyncio.create_task(
        asyncio.to_thread(
            process_extraction,
            request.text,
            request.confidence_threshold,
            request.include_regex,
        )
    )
    release_in_finally = True
    try:
        if REQUEST_TIMEOUT_SECONDS > 0:
            try:
                return await asyncio.wait_for(
                    asyncio.shield(worker_task),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except TimeoutError as exc:
                release_in_finally = False
                worker_task.add_done_callback(_release_slot_after_timeout)
                raise HTTPException(
                    status_code=504,
                    detail="NER inference timed out",
                ) from exc
        return await worker_task
    finally:
        if release_in_finally:
            _infer_slots.release()


@app.get("/health")
async def health_check():
    if ner_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    payload: dict[str, Any] = {
        "status": "ok",
        "busy": _infer_slots.locked(),
    }
    if EXPOSE_HEALTH_CONFIG:
        payload.update(
            {
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "aggregation": AGGREGATION,
                "max_length": MAX_LENGTH,
                "stride": STRIDE,
                "max_input_chars": MAX_INPUT_CHARS,
                "max_in_flight": MAX_IN_FLIGHT,
                "jurisprudencia_ner_threshold": JURISPRUDENCIA_NER_THRESHOLD,
            }
        )
    return payload


@app.post("/api/v1/extract", response_model=ExtractResponse)
async def extract_entities(request: ExtractRequest):
    if ner_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        entities = await _run_extraction_with_slot(request)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Erro na inferência") from e

    processing_time_ms = int((time.time() - start_time) * 1000)
    return ExtractResponse(
        status="success",
        processing_time_ms=processing_time_ms,
        entities=entities,
    )
