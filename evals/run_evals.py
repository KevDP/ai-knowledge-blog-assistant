"""
run_evals.py - Regression testing for the EVA agent (Phase 2).

Runs a golden set of queries against the LangGraph pipeline and verifies its
behavior according to decisions (routing, off-topic gating, retrieval recall, memory).
That is deliberate:

  - The agent's decisions are what regresses when I tweak a threshold, a
    reformulation prompt, or the router. They are (mostly) deterministic and
    cheap to verify.
  - Judging answer quality would require an LLM-as-judge (costly and non-deterministic).

Each case can be:
  - "question": a single-turn question
  - "conversation": a list of user messages replayed in order to build real
    history; the LAST turn is asserted (memory test).

Usage:
    python -m evals.run_evals
    python -m evals.run_evals --verbose

Cost note: this invokes the LLM (answer + utility calls). With ~13 cases it
costs a few cents. Utility-call cost is reported at the end; the answer()
cost is logged separately to stderr from llm.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.agent import run_turn
from src.llm import CANNED_OFF_TOPIC

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.json"


def _is_off_topic(result: dict) -> bool:
    """A turn is off-topic if the graph returned the canned response."""
    return result.get("answer", "").strip() == CANNED_OFF_TOPIC.strip()


def _check(result: dict, expect: dict) -> list[str]:
    """Returns the list of failures (empty = passed)."""
    failures: list[str] = []

    if "off_topic" in expect:
        got = _is_off_topic(result)
        if got != expect["off_topic"]:
            failures.append(f"off_topic: expected {expect['off_topic']}, got {got}")

    if "route" in expect:
        got = result.get("route")
        if got != expect["route"]:
            failures.append(f"route: expected '{expect['route']}', got '{got}'")

    if "grade" in expect:
        got = result.get("grade")
        if got != expect["grade"]:
            failures.append(f"grade: expected '{expect['grade']}', got '{got}'")

    if "sources_include" in expect:
        # Correct RAG assertion: did the right doc ENTER the top-k (context window)
        # We do not require rank #1
        # What matters for groundedness is that the doc is available to the LLM.
        sources = result.get("sources") or []
        got = [s["source"] for s in sources]
        if expect["sources_include"] not in got:
            failures.append(
                f"sources_include: expected '{expect['sources_include']}' in {got}"
            )

    if "min_score" in expect:
        got = result.get("best_score", 0.0)
        if got < expect["min_score"]:
            failures.append(f"min_score: expected ≥{expect['min_score']}, got {got:.3f}")

    return failures


def _run_case(case: dict) -> dict:
    """
    Replays the case (single turn or conversation) and returns the result of
    the LAST turn plus the accumulated utility cost.
    """
    total_util_cost = 0.0

    if "conversation" in case:
        history: list[dict] = []
        result: dict = {}
        for msg in case["conversation"]:
            result = run_turn(msg, history=history)
            total_util_cost += result.get("utility_cost_usd", 0.0)
            history = history + [
                {"role": "user", "content": msg},
                {"role": "assistant",
                 "content": result.get("answer", ""),
                 "sources": result.get("sources", [])},
            ]
    else:
        result = run_turn(case["question"])
        total_util_cost += result.get("utility_cost_usd", 0.0)

    result["_util_cost"] = total_util_cost
    return result


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="EVA agent eval runner")
    parser.add_argument("--verbose", action="store_true", help="Print the answer for each case")
    args = parser.parse_args()

    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    passed = 0
    known_gaps = 0
    unexpected = 0
    total_cost = 0.0
    print(f"\nRunning {len(cases)} eval cases...\n")

    for case in cases:
        result = _run_case(case)
        total_cost += result.get("_util_cost", 0.0)
        failures = _check(result, case.get("expect", {}))

        if not failures:
            passed += 1
            print(f"  Passed {case['id']}")
        elif case.get("known_limitation"):
            # this is expected as a documented failure.
            known_gaps += 1
            print(f"  Not passed {case['id']}  (known limitation)")
            print(f"    {case['known_limitation']}")
            for f in failures:
                print(f"      - {f}")
        else:
            unexpected += 1
            print(f"  ✗ {case['id']}")
            for f in failures:
                print(f"      - {f}")

        if args.verbose:
            print(f"      A: {result.get('answer', '')[:120]}")

    print(f"\n{'─' * 48}")
    print(f"  {passed} passed · {known_gaps} known limitation(s) · "
          f"{unexpected} unexpected failure(s)")
    print(f"  utility LLM cost: ${total_cost:.6f}")
    print(f"  (answer() cost logged separately to stderr)\n")

    # The suite passes if there are no unexpected failures. 
    # Known limitations do not count as a failure, they are tracked for a planned fix in the KB.
    raise SystemExit(0 if unexpected == 0 else 1)


if __name__ == "__main__":
    main()
