"""Score NER API output against benchmark expectations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Expectation:
    label: str
    text: str


@dataclass
class BenchmarkCase:
    id: str
    description: str
    text: str
    must_contain: list[Expectation] = field(default_factory=list)
    must_not_contain: list[Expectation] = field(default_factory=list)
    regex_only_must: list[Expectation] = field(default_factory=list)


@dataclass
class CaseScore:
    case_id: str
    passed: bool
    hits: int
    misses: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)
    entity_count: int = 0
    by_label: dict[str, int] = field(default_factory=dict)


def load_cases(path: Path | None = None) -> list[BenchmarkCase]:
    path = path or Path(__file__).with_name("cases.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[BenchmarkCase] = []
    for item in raw:
        cases.append(
            BenchmarkCase(
                id=item["id"],
                description=item.get("description", ""),
                text=item["text"],
                must_contain=[
                    Expectation(label=e["label"], text=e["text"])
                    for e in item.get("must_contain", [])
                ],
                must_not_contain=[
                    Expectation(label=e["label"], text=e["text"])
                    for e in item.get("must_not_contain", [])
                ],
                regex_only_must=[
                    (
                        Expectation(label=e["label"], text=e.get("text", ""))
                        if isinstance(e, dict)
                        else Expectation(label=str(e), text="")
                    )
                    for e in item.get("regex_only_must", [])
                ],
            )
        )
    return cases


def _norm(value: str) -> str:
    return value.casefold().strip()


def _entity_rows(entities: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for ent in entities:
        label = str(ent.get("entity_group") or ent.get("label") or "").upper()
        text = str(ent.get("text") or ent.get("word") or "").strip()
        source = str(ent.get("source") or "ner")
        if label and text:
            rows.append((label, text, source))
    return rows


def _matches(expect: Expectation, rows: list[tuple[str, str, str]]) -> bool:
    want_label = expect.label.upper()
    needle = _norm(expect.text)
    for label, text, _source in rows:
        if label == want_label and needle in _norm(text):
            return True
    return False


def _false_positive(expect: Expectation, rows: list[tuple[str, str, str]]) -> bool:
    want_label = expect.label.upper()
    needle = _norm(expect.text)
    for label, text, _source in rows:
        if label == want_label and needle in _norm(text):
            return True
    return False


def score_case(case: BenchmarkCase, entities: list[dict[str, Any]]) -> CaseScore:
    rows = _entity_rows(entities)
    by_label: dict[str, int] = {}
    for label, _text, _src in rows:
        by_label[label] = by_label.get(label, 0) + 1

    misses: list[str] = []
    for expect in case.must_contain:
        if not _matches(expect, rows):
            misses.append(f"missing {expect.label} ~{expect.text!r}")

    false_positives: list[str] = []
    for expect in case.must_not_contain:
        if _false_positive(expect, rows):
            false_positives.append(f"unwanted {expect.label} ~{expect.text!r}")

    passed = not misses and not false_positives
    hits = len(case.must_contain) - len(misses)
    return CaseScore(
        case_id=case.id,
        passed=passed,
        hits=hits,
        misses=misses,
        false_positives=false_positives,
        entity_count=len(rows),
        by_label=by_label,
    )


def score_regex_only_labels(case: BenchmarkCase, entities: list[dict[str, Any]]) -> list[str]:
    if not case.regex_only_must:
        return []
    rows = _entity_rows(entities)
    missing: list[str] = []
    for expect in case.regex_only_must:
        if isinstance(expect, str):
            expect = Expectation(label=expect, text="")
        want_label = expect.label.upper()
        needle = _norm(expect.text)
        if not any(
            label == want_label
            and source == "regex"
            and (not needle or needle in _norm(text))
            for label, text, source in rows
        ):
            suffix = f" ~{expect.text!r}" if expect.text else ""
            missing.append(f"regex missing {expect.label}{suffix}")
    return missing


def summarize_scores(scores: list[CaseScore]) -> dict[str, Any]:
    total = len(scores)
    passed = sum(1 for s in scores if s.passed)
    return {
        "cases": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "total_entities": sum(s.entity_count for s in scores),
    }
