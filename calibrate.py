#!/usr/bin/env python3
"""
Varre threshold e include_regex contra benchmarks/cases.json.

Uso:
  python calibrate.py
  python calibrate.py --endpoint http://192.168.1.221:5522/api/v1/extract
  python calibrate.py --thresholds 0.3,0.4,0.5,0.6 --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

from benchmarks.scoring import (
    CaseScore,
    load_cases,
    score_case,
    score_regex_only_labels,
    summarize_scores,
)

DEFAULT_ENDPOINT = "http://127.0.0.1:5522/api/v1/extract"


def call_ner(
    endpoint: str,
    text: str,
    *,
    threshold: float,
    include_regex: bool,
    timeout: int,
) -> tuple[list[dict], int]:
    t0 = time.perf_counter()
    resp = requests.post(
        endpoint,
        json={
            "text": text,
            "confidence_threshold": threshold,
            "include_regex": include_regex,
        },
        timeout=timeout,
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    resp.raise_for_status()
    data = resp.json()
    return list(data.get("entities") or []), elapsed_ms


def run_grid(
    endpoint: str,
    thresholds: list[float],
    include_regex_options: list[bool],
    timeout: int,
) -> list[dict]:
    cases = load_cases()
    results: list[dict] = []

    for include_regex in include_regex_options:
        for threshold in thresholds:
            scores: list[CaseScore] = []
            latencies: list[int] = []
            regex_gaps: list[str] = []

            for case in cases:
                entities, ms = call_ner(
                    endpoint,
                    case.text,
                    threshold=threshold,
                    include_regex=include_regex,
                    timeout=timeout,
                )
                latencies.append(ms)
                scores.append(score_case(case, entities))
                regex_gaps.extend(score_regex_only_labels(case, entities))

            summary = summarize_scores(scores)
            results.append(
                {
                    "threshold": threshold,
                    "include_regex": include_regex,
                    **summary,
                    "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
                    "regex_gaps": regex_gaps,
                    "failures": [
                        {
                            "case": s.case_id,
                            "misses": s.misses,
                            "false_positives": s.false_positives,
                        }
                        for s in scores
                        if not s.passed
                    ],
                }
            )
            print(
                f"threshold={threshold:.2f} regex={include_regex} "
                f"pass={summary['passed']}/{summary['cases']} "
                f"entities={summary['total_entities']} "
                f"avg_ms={results[-1]['avg_latency_ms']}"
            )
            if regex_gaps:
                print(f"  regex gaps: {', '.join(regex_gaps[:5])}")

    return results


def pick_best(results: list[dict]) -> dict | None:
    ok = [r for r in results if r["pass_rate"] == 1.0]
    if not ok:
        ok = sorted(results, key=lambda r: (-r["pass_rate"], r["total_entities"]))[:1]
    if not ok:
        return None
    return sorted(
        ok,
        key=lambda r: (-r["include_regex"], -r["threshold"], r["avg_latency_ms"]),
    )[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Legal NER thresholds")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--include-regex", choices=("true", "false", "both"), default="both")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    if args.include_regex == "both":
        regex_opts = [False, True]
    else:
        regex_opts = [args.include_regex == "true"]

    try:
        health = requests.get(args.endpoint.replace("/api/v1/extract", "/health"), timeout=10)
        health.raise_for_status()
        print("health:", health.json())
    except requests.RequestException as exc:
        print(f"health check failed: {exc}", file=sys.stderr)
        return 1

    results = run_grid(args.endpoint, thresholds, regex_opts, args.timeout)
    best = pick_best(results)
    if best:
        print("\nRecomendado:")
        print(
            json.dumps(
                {
                    "confidence_threshold": best["threshold"],
                    "include_regex": best["include_regex"],
                    "pass_rate": best["pass_rate"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if args.output:
        args.output.write_text(
            json.dumps({"runs": results, "recommended": best}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nRelatório salvo em {args.output}")

    return 0 if best and best.get("pass_rate", 0) >= 0.75 else 1


if __name__ == "__main__":
    raise SystemExit(main())
