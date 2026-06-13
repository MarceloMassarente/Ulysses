"""Unit tests for NER helpers (no model load)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from main import (  # noqa: E402
    clean_entity_text,
    is_noisy_entity,
    jurisprudence_regex_entities,
    merge_predictions,
    normalize_legal_text,
    normalize_pdf_text,
    process_batch_extraction,
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
    assert len(out) == 2
    assert all(e.text == "João da Silva" for e in out)
    assert {e.start for e in out} == {10, 50}


def test_merge_predictions_dedups_exact_same_span() -> None:
    preds = [
        {
            "entity_group": "PESSOA",
            "word": "João da Silva",
            "score": 0.7,
            "start": 10,
            "end": 23,
        },
        {
            "entity_group": "PESSOA",
            "word": "João da Silva",
            "score": 0.9,
            "start": 10,
            "end": 23,
        },
    ]
    out = merge_predictions(preds, threshold=0.5)
    assert len(out) == 1
    assert out[0].score == 0.9


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


def test_filters_oab_as_pessoa() -> None:
    assert is_noisy_entity("PESSOA", "OAB/SP 123.456")
    assert is_noisy_entity("PESSOA", "OAB")


def test_regex_tema_and_apelacao() -> None:
    text = "Tema 810 do STF. Apelacao Civel n 1234567-89.2020.8.26.0100."
    found = regex_entities(text, 0.0)
    words = [e["word"] for e in found if e["entity_group"] == "JURISPRUDENCIA"]
    assert any("Tema 810" in w for w in words)
    assert any("Apelacao" in w or "Apela" in w for w in words)


def test_regex_advogado_comarca_data_valor() -> None:
    text = (
        "Dr. Carlos Mendes, OAB/SP 123.456. "
        "Comarca de Campinas. "
        "Sentença em 15 de março de 2024. "
        "Condenação de 10.000,00 reais."
    )
    found = regex_entities(text, 0.0)
    by_label = {e["entity_group"]: e["word"] for e in found}
    assert "Carlos" in by_label.get("PESSOA", "")
    assert "Campinas" in by_label.get("LOCAL", "")
    assert any(e["entity_group"] == "DATA" and "15/03" in e["word"] for e in found)
    assert any(e["entity_group"] == "VALOR" and "reais" in e["word"].lower() for e in found)


def test_regex_doutrina() -> None:
    text = "Conforme Marinoni, 2020, p. 45, e doutrina de Humberto Theodoro."
    found = regex_entities(text, 0.0)
    doutrina = [e["word"] for e in found if e["entity_group"] == "DOUTRINA"]
    assert any("Marinoni" in w for w in doutrina)


def test_normalize_pdf_rejoins_broken_org() -> None:
    raw = "BAN\nCO XPTO S.A."
    norm = normalize_pdf_text(raw)
    assert "BANCO" in norm
    found = regex_entities(norm, 0.0)
    orgs = [e["word"] for e in found if e["entity_group"] == "ORGANIZACAO"]
    assert any("XPTO" in w for w in orgs)


def test_normalize_pdf_preserves_uppercase_connectors() -> None:
    raw = "BANCO\nDO\nBRASIL S.A."
    norm = normalize_pdf_text(raw)
    assert "BANCO DO BRASIL" in norm
    assert "BANCODO" not in norm


def test_jurisprudencia_marker_requires_word_boundary() -> None:
    assert is_noisy_entity("JURISPRUDENCIA", "parecer tecnico", source="ner")


def test_regex_recurso_requires_formal_number() -> None:
    found = regex_entities("recurso 2024 e Embargos 2", 0.0)
    words = [e["word"] for e in found if e["entity_group"] == "JURISPRUDENCIA"]
    assert not words


def test_regex_common_unformatted_ids_and_lei_numero() -> None:
    text = "CPF 12345678901 CNPJ 12345678000190 Lei nº 8.666/1993"
    found = regex_entities(text, 0.0)
    labels = {e["entity_group"] for e in found}
    assert {"CPF", "CNPJ", "LEGISLACAO"} <= labels


def test_process_batch_extraction_uses_pipeline_batch(monkeypatch) -> None:
    calls = []

    class Tokenizer:
        is_fast = True

    class Pipe:
        tokenizer = Tokenizer()

        def __call__(self, texts, stride=None):
            calls.append((texts, stride))
            return [
                [{"entity_group": "PESSOA", "word": "Ana", "score": 0.9}],
                [{"entity_group": "PESSOA", "word": "Bruno", "score": 0.8}],
            ]

    monkeypatch.setattr(main, "ner_pipeline", Pipe())

    out = process_batch_extraction(["Ana peticionou.", "Bruno contestou."], 0.5, False)

    assert len(out) == 2
    assert out[0][0].text == "Ana"
    assert out[1][0].text == "Bruno"
    assert len(calls) == 1
    assert calls[0][0] == ["Ana peticionou.", "Bruno contestou."]


def test_extract_batch_endpoint_returns_indexed_results(monkeypatch) -> None:
    class Tokenizer:
        is_fast = True

    class Pipe:
        tokenizer = Tokenizer()

        def __call__(self, texts, stride=None):
            return [
                [{"entity_group": "PESSOA", "word": "Ana", "score": 0.9}],
                [{"entity_group": "PESSOA", "word": "Bruno", "score": 0.8}],
            ]

    monkeypatch.setattr(main, "ner_pipeline", Pipe())

    response = TestClient(main.app).post(
        "/api/v1/extract_batch",
        json={
            "texts": ["Ana peticionou.", "Bruno contestou."],
            "confidence_threshold": 0.5,
            "include_regex": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert [item["index"] for item in body["results"]] == [0, 1]
    assert body["results"][0]["entities"][0]["text"] == "Ana"
    assert body["results"][1]["entities"][0]["text"] == "Bruno"


def test_process_extraction_fills_metrics(monkeypatch) -> None:
    class Tokenizer:
        is_fast = True

    class Pipe:
        tokenizer = Tokenizer()

        def __call__(self, text, stride=None):
            return [{"entity_group": "PESSOA", "word": "Ana", "score": 0.9}]

    monkeypatch.setattr(main, "ner_pipeline", Pipe())

    metrics: dict[str, object] = {}
    out = main.process_extraction("Ana peticionou.", 0.0, True, metrics)

    assert metrics["batch_size"] == 1
    assert metrics["entities"] == len(out)
    for key in ("normalize_ms", "infer_ms", "regex_ms", "postprocess_ms"):
        assert isinstance(metrics[key], float)


def test_process_batch_extraction_fills_metrics(monkeypatch) -> None:
    class Tokenizer:
        is_fast = True

    class Pipe:
        tokenizer = Tokenizer()

        def __call__(self, texts, stride=None):
            return [
                [{"entity_group": "PESSOA", "word": "Ana", "score": 0.9}],
                [{"entity_group": "PESSOA", "word": "Bruno", "score": 0.8}],
            ]

    monkeypatch.setattr(main, "ner_pipeline", Pipe())

    metrics: dict[str, object] = {}
    out = main.process_batch_extraction(["Ana peticionou.", "Bruno contestou."], 0.0, False, metrics)

    assert metrics["batch_size"] == 2
    assert metrics["entities"] == sum(len(item) for item in out)
    assert isinstance(metrics["infer_ms"], float)


def test_health_reports_in_flight_and_counters(monkeypatch) -> None:
    monkeypatch.setattr(main, "ner_pipeline", object())

    body = TestClient(main.app).get("/health").json()

    assert body["status"] == "ok"
    assert body["in_flight"] == 0
    assert {"requests_total", "batch_requests_total", "errors_503", "errors_504"} <= set(
        body["requests"]
    )
