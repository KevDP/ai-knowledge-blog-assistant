"""
run_evals.py — Regression testing del agente EVA (Fase 2).

Corre un golden set de queries contra el grafo LangGraph y verifica sus
DECISIONES (routing, off-topic gating, top source, score) — no la calidad
prosística de la respuesta. Eso es deliberado:

  - Las decisiones del agente son lo que regresiona cuando toco un umbral,
    un prompt de reformulación, o el router. Son deterministas-ish y baratas
    de verificar.
  - Juzgar calidad de la respuesta requeriría un LLM-as-judge (costoso, no
    determinista). Eso queda para una Parte 3, si hace falta.

Cada caso puede ser:
  - "question": una sola pregunta (turno único), o
  - "conversation": lista de mensajes de usuario; se reproducen en orden para
    construir historial real, y se verifica el ÚLTIMO turno (prueba de memoria).

Uso:
    python -m evals.run_evals
    python -m evals.run_evals --verbose

Nota de costo: correr esto invoca al LLM (answer + utilitarias). Con ~12 casos
son unos centavos. El costo de las llamadas utilitarias se reporta al final;
el de answer() se loggea por separado a stderr desde llm.py.
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
    """Un turno es off-topic si el grafo devolvió la respuesta enlatada."""
    return result.get("answer", "").strip() == CANNED_OFF_TOPIC.strip()


def _check(result: dict, expect: dict) -> list[str]:
    """Devuelve la lista de fallas (vacía = pasó)."""
    failures: list[str] = []

    if "off_topic" in expect:
        got = _is_off_topic(result)
        if got != expect["off_topic"]:
            failures.append(f"off_topic: esperaba {expect['off_topic']}, obtuve {got}")

    if "route" in expect:
        got = result.get("route")
        if got != expect["route"]:
            failures.append(f"route: esperaba '{expect['route']}', obtuve '{got}'")

    if "grade" in expect:
        got = result.get("grade")
        if got != expect["grade"]:
            failures.append(f"grade: esperaba '{expect['grade']}', obtuve '{got}'")

    if "sources_include" in expect:
        # Assertion de RAG correcta: ¿el doc correcto ENTRÓ al top-k (context
        # window)? No exigimos que sea el #1 — que sea el ranking exacto es una
        # nuance de precisión (territorio de reranking, Parte 3). Lo que importa
        # para groundedness es que esté disponible para el LLM.
        sources = result.get("sources") or []
        got = [s["source"] for s in sources]
        if expect["sources_include"] not in got:
            failures.append(
                f"sources_include: esperaba '{expect['sources_include']}' en {got}"
            )

    if "min_score" in expect:
        got = result.get("best_score", 0.0)
        if got < expect["min_score"]:
            failures.append(f"min_score: esperaba ≥{expect['min_score']}, obtuve {got:.3f}")

    return failures


def _run_case(case: dict) -> dict:
    """Reproduce el caso (turno único o conversación) y devuelve el resultado
    del ÚLTIMO turno más el costo utilitario acumulado."""
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
    parser.add_argument("--verbose", action="store_true", help="Imprime respuesta de cada caso")
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
            print(f"  ✓ {case['id']}")
        elif case.get("known_limitation"):
            # Falla ESPERADA y documentada. No es una regresión — es un gap
            # conocido con un fix planeado. Un eval maduro distingue "esto se
            # rompió" de "esto todavía no lo arreglamos".
            known_gaps += 1
            print(f"  ⚠ {case['id']}  (known limitation)")
            print(f"      → {case['known_limitation']}")
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

    # La suite pasa si no hay fallas INESPERADAS. Los known limitations no
    # cuentan como regresión — están rastreados para Parte 3 (reranking).
    raise SystemExit(0 if unexpected == 0 else 1)


if __name__ == "__main__":
    main()
