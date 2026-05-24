"""Unit tests for NER helpers (no model load)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import (  # noqa: E402
    clean_entity_text,
    is_noisy_entity,
    merge_predictions,
    normalize_pdf_text,
    regex_entities,
)


def test_clean_entity_text_fixes_spacing() -> None:
    assert clean_entity_text("8 . 666 / 1993") == "8.666/1993"


def test_is_noisy_entity_filters_tempo_fragments() -> None:
    assert is_noisy_entity("TEMPO", "05")
    assert is_noisy_entity("TEMPO", "202")
    assert not is_noisy_entity("PESSOA", "João da Silva")


def test_merge_predictions_dedup_and_text_field() -> None:
    preds = [
        {
            "entity_group": "PESSOA",
            "word": "João da Silva",
            "score": 0.9,
            "start": 10,
            "end": 23,
        },
        {
            "entity_group": "PESSOA",
            "word": "João da Silva",
            "score": 0.7,
            "start": 50,
            "end": 63,
        },
    ]
    out = merge_predictions(preds, threshold=0.5)
    assert len(out) == 1
    assert out[0].text == "João da Silva"
    assert out[0].score == 0.9
    assert out[0].start == 10


def test_regex_entities_finds_cnpj() -> None:
    text = "Empresa CNPJ 12.345.678/0001-90 no processo."
    found = regex_entities(text, 0.0)
    labels = {e["entity_group"] for e in found}
    assert "CNPJ" in labels


def test_normalize_pdf_text() -> None:
    raw = "linha um\nlinha dois\n\nparágrafo"
    norm = normalize_pdf_text(raw)
    assert "linha um linha dois" in norm
