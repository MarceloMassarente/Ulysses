#!/usr/bin/env python3
"""Testa casos jurídicos típicos contra a API NER."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

ENDPOINT = os.environ.get(
    "LEGAL_NER_ENDPOINT", "http://192.168.1.221:5522/api/v1/extract"
)
THRESHOLD = float(os.environ.get("LEGAL_NER_THRESHOLD", "0.5"))
INCLUDE_REGEX = os.environ.get("LEGAL_NER_INCLUDE_REGEX", "true").lower() in (
    "1",
    "true",
    "yes",
)


def load_cases() -> list[dict]:
    path = Path(__file__).parent / "benchmarks" / "edge_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def call_api(text: str) -> list[dict]:
    r = requests.post(
        ENDPOINT,
        json={
            "text": text,
            "confidence_threshold": THRESHOLD,
            "include_regex": INCLUDE_REGEX,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("entities") or []


def check_expectations(entities: list[dict], expects: list[dict]) -> tuple[bool, list[str]]:
    misses: list[str] = []
    rows = [
        (
            str(e.get("entity_group", "")).upper(),
            str(e.get("text") or e.get("word", "")),
            str(e.get("source", "?")),
        )
        for e in entities
    ]
    for exp in expects:
        label = exp["label"].upper()
        needle = exp["contains"]
        if not any(
            lab == label and needle.lower() in txt.lower()
            for lab, txt, _src in rows
        ):
            misses.append(f"falta {label} ~'{needle}'")
    return (not misses, misses)


def main() -> int:
    try:
        health = requests.get(ENDPOINT.replace("/api/v1/extract", "/health"), timeout=10)
        health.raise_for_status()
        print("health:", health.json())
    except requests.RequestException as exc:
        print(f"ERRO health: {exc}", file=sys.stderr)
        return 1

    cases = load_cases()
    by_cat: dict[str, list] = {}
    passed = 0
    report: list[dict] = []

    print(f"\nendpoint={ENDPOINT} threshold={THRESHOLD} regex={INCLUDE_REGEX}\n")
    print(f"{'ID':<28} {'CAT':<14} {'OK':<4} ENT  DETALHE")
    print("-" * 90)

    for case in cases:
        entities = call_api(case["text"])
        ok, misses = check_expectations(entities, case.get("expect", []))
        report.append(
            {
                "id": case["id"],
                "category": case.get("category"),
                "passed": ok,
                "misses": misses,
                "entities": entities,
            }
        )
        if ok:
            passed += 1
        cat = case.get("category", "?")
        by_cat.setdefault(cat, []).append((case["id"], ok))
        status = "sim" if ok else "NAO"
        detail = "" if ok else "; ".join(misses)
        ent_preview = ", ".join(
            f"{e['entity_group']}:{(e.get('text') or '')[:25]}"
            for e in entities[:4]
        )
        if len(entities) > 4:
            ent_preview += f" (+{len(entities)-4})"
        print(
            f"{case['id']:<28} {cat:<14} {status:<4} {len(entities):<3} "
            f"{detail or ent_preview}"
        )

    print("-" * 90)
    print(f"Total: {passed}/{len(cases)} passaram ({100*passed/len(cases):.0f}%)\n")

    print("Por categoria:")
    for cat, items in sorted(by_cat.items()):
        n = sum(1 for _id, ok in items if ok)
        print(f"  {cat:<14} {n}/{len(items)}")

    out = Path(__file__).parent / "benchmarks" / "edge_cases_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRelatório: {out}")
    return 0 if passed >= len(cases) * 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
