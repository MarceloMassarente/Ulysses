"""Offline tests for benchmark scoring (no model/API)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.scoring import BenchmarkCase, Expectation, load_cases, score_case, score_regex_only_labels  # noqa: E402


def test_load_cases_has_entries() -> None:
    cases = load_cases()
    assert len(cases) >= 5
    assert any(c.id == "peticao_hibrida" for c in cases)


def test_score_case_passes_when_expectations_met() -> None:
    case = BenchmarkCase(
        id="x",
        description="",
        text="t",
        must_contain=[Expectation(label="PROCESSO", text="1001234")],
        must_not_contain=[Expectation(label="TEMPO", text="05")],
    )
    entities = [
        {"entity_group": "PROCESSO", "text": "1001234-56.2024.8.26.0100", "source": "regex"},
        {"entity_group": "TEMPO", "text": "2024", "source": "ner"},
    ]
    result = score_case(case, entities)
    assert result.passed
    assert result.hits == 1


def test_score_case_fails_on_fragment() -> None:
    case = BenchmarkCase(
        id="x",
        description="",
        text="t",
        must_not_contain=[Expectation(label="ORGANIZACAO", text="CO")],
    )
    entities = [{"entity_group": "ORGANIZACAO", "text": "CO", "source": "ner"}]
    result = score_case(case, entities)
    assert not result.passed
    assert result.false_positives


def test_regex_only_must_detects_missing_source() -> None:
    case = BenchmarkCase(
        id="x",
        description="",
        text="t",
        regex_only_must=["PROCESSO"],
    )
    entities = [{"entity_group": "PROCESSO", "text": "123", "source": "ner"}]
    gaps = score_regex_only_labels(case, entities)
    assert gaps
