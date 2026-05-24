"""Unit tests for NER helpers (no model load)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import (  # noqa: E402
    clean_entity_text,
    is_noisy_entity,
    jurisprudence_regex_entities,
    merge_predictions,
    normalize_legal_text,
    normalize_pdf_text,
    reclassify_ner_predictions,
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


def test_normalize_legal_text_fixes_sumula_typo() -> None:
    assert "Súmula 7" in normalize_legal_text("Conforme Sumula 7 do STJ")


def test_jurisprudence_regex_sumula_and_resp() -> None:
    text = "Sumula 7 do STJ e REsp 1.234.567/SP aplicam-se."
    found = jurisprudence_regex_entities(normalize_legal_text(text))
    labels = {e["word"] for e in found}
    assert any("Súmula 7" in w for w in labels)
    assert any("REsp" in w for w in labels)


def test_reclassify_legislacao_to_jurisprudencia() -> None:
    preds = [
        {"entity_group": "LEGISLACAO", "word": "Súmula 54 do STJ", "score": 0.8},
        {"entity_group": "LEGISLACAO", "word": "art. 300", "score": 0.8},
    ]
    out = reclassify_ner_predictions(preds)
    assert out[0]["entity_group"] == "JURISPRUDENCIA"
    assert out[1]["entity_group"] == "LEGISLACAO"


def test_jurisprudencia_ner_lower_threshold() -> None:
    preds = [
        {
            "entity_group": "JURISPRUDENCIA",
            "word": "REsp 1.729.593/SP",
            "score": 0.42,
            "source": "ner",
        }
    ]
    out = merge_predictions(preds, threshold=0.5)
    assert len(out) == 1


def test_filters_cnj_as_jurisprudencia() -> None:
    assert is_noisy_entity("JURISPRUDENCIA", "1010761-40.2024.8.26.0032-TJSP")


def test_jurisprudence_regex_accented_double_sumula() -> None:
    text = "Súmula 7 do STJ e Súmula Vinculante 11 do STF"
    found = jurisprudence_regex_entities(text)
    words = [e["word"] for e in found]
    assert any("Súmula 7" in w for w in words)
    assert any("Vinculante" in w or "11" in w for w in words)


def test_filters_juris_ner_noise() -> None:
    assert is_noisy_entity("JURISPRUDENCIA", "2024", source="ner")
    assert is_noisy_entity("JURISPRUDENCIA", "mula", source="ner")
    assert not is_noisy_entity("JURISPRUDENCIA", "REsp 1.234.567/SP", source="ner")
