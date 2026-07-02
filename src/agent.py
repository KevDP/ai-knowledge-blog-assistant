"""
agent.py — Capa de orquestación LangGraph sobre el core RAG de EVA (Fase 2).

Envuelve los retrieve() + answer() existentes con:
  - Memoria de conversación (historial que se pasa por turno)
  - Routing híbrido (heurístico primero, LLM solo en casos ambiguos)
  - Reformulación de query cuando el retrieval sale débil (capada, con guardia de costo)
  - Manejo de meta-queries diagnósticas ("show sources", "más corto")

Principio de diseño: este archivo ORQUESTA. retrieve.py y llm.py siguen
siendo el motor. Aquí no se reimplementa retrieval ni generación — se decide
QUÉ hacer y EN QUÉ ORDEN.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Literal, TypedDict

from anthropic import Anthropic
from langgraph.graph import END, START, StateGraph

from src.llm import (
    CANNED_OFF_TOPIC,
    RELEVANCE_THRESHOLD as OFF_TOPIC_THRESHOLD,  # 0.55 en Fase 0 (BGE)
    answer as generate_answer,
)
from src.retrieve import retrieve

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de control de flujo (y de costo)
# ─────────────────────────────────────────────────────────────────────────────

# Banda de scores (escala BGE, Fase 0). OJO: al migrar a Titan (Fase 1) estos
# umbrales SE RECALIBRAN — los rangos de similitud coseno son específicos del
# embedder, como expliqué en la Parte 1 del blog.
#   score < OFF_TOPIC_THRESHOLD        → off-topic  → canned ($0)
#   OFF_TOPIC_THRESHOLD ≤ score < WEAK → débil      → reformula (1 intento)
#   score ≥ WEAK                       → bueno      → responde
WEAK_BAND_CEIL = 0.62

# Guardia anti-loop = guardia anti-costo. Sin esto, reformula→retrieve→grade
# se vuelve un ataque de costo contra uno mismo.
MAX_RETRIES = 1

# Modelo para las llamadas "utilitarias" (coref, reformulación, fallback de
# routing). Haiku porque son prompts cortos y baratos.
UTILITY_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
UTILITY_MAX_TOKENS = 80

# Heurística de meta-queries: palabras que delatan una pregunta diagnóstica
# (sobre las fuentes) o de reformato (más corto). Regex, $0.
_META_SOURCES_RE = re.compile(
    r"\b(sources?|fuentes?|cita|citar|de d[oó]nde|where.*from|show.*source)\b",
    re.IGNORECASE,
)
_META_CONCISE_RE = re.compile(
    r"\b(shorter|m[aá]s corto|m[aá]s breve|briefly|concise|res[uú]mel[oa])\b",
    re.IGNORECASE,
)

# Heurística de correferencia: pronombres colgantes que sugieren que la pregunta
# depende del turno anterior ("what did HE study?").
_PRONOUN_RE = re.compile(
    r"\b(he|she|it|they|him|her|that|there|this|"
    r"[ée]l|ella|eso|esa|ese|ah[ií]|ahi|su|sus)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Estado del grafo
# ─────────────────────────────────────────────────────────────────────────────
class EvaState(TypedDict, total=False):
    question: str          # pregunta cruda del usuario
    history: list[dict]    # turnos previos → MEMORIA. [{role, content, sources?}]
    query: str             # query resuelta/reformulada usada para retrieval
    retrieved: list        # chunks de retrieve()
    best_score: float      # score del top chunk
    route: str             # "meta" | "normal" (fijado por route_question)
    grade: str             # "answer" | "reformulate" | "off_topic"
    retries: int           # contador de reintentos (cap = MAX_RETRIES)
    answer: str            # respuesta final
    sources: list          # fuentes de los chunks usados
    utility_cost_usd: float  # costo acumulado de llamadas utilitarias del turno


# ─────────────────────────────────────────────────────────────────────────────
# Helper: llamada LLM utilitaria (coref / reformulación / fallback de routing)
# ─────────────────────────────────────────────────────────────────────────────
_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY no configurada.")
        _client = Anthropic(api_key=api_key)
    return _client


def _utility_llm(prompt: str, *, max_tokens: int = UTILITY_MAX_TOKENS) -> tuple[str, float]:
    """Llamada corta y barata. Regresa (texto, costo_estimado_usd)."""
    resp = _get_client().messages.create(
        model=UTILITY_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    u = resp.usage
    # Mismos precios placeholder que llm.py — ajustar a pricing real de Bedrock.
    cost = (u.input_tokens * 1.0 + u.output_tokens * 5.0) / 1_000_000
    return resp.content[0].text.strip(), cost


def _fmt_history(history: list[dict], *, last_n: int = 4) -> str:
    """Serializa los últimos turnos para dar contexto al LLM."""
    turns = history[-last_n:]
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


# ─────────────────────────────────────────────────────────────────────────────
# NODOS
# ─────────────────────────────────────────────────────────────────────────────
def route_question(state: EvaState) -> dict:
    """
    Router HÍBRIDO. Heurística primero ($0); LLM solo si el heurístico duda.

    - Meta-query (fuentes / más corto) detectable por regex → route="meta".
    - Todo lo demás → route="normal".
    - Fallback LLM: solo cuando la query es corta y ambigua CON historial
      presente (podría ser meta implícita). Es el único caso que paga routing.
    """
    q = state["question"]
    history = state.get("history", [])

    if _META_SOURCES_RE.search(q) or _META_CONCISE_RE.search(q):
        return {"route": "meta"}

    # Ambigüedad: query muy corta (≤3 palabras) con historial. Puede ser meta
    # implícita ("y las fuentes?") o follow-up normal. Aquí sí vale un LLM.
    if history and len(q.split()) <= 3:
        prompt = (
            "Classify the user's message as META (a request about the previous "
            "answer's sources or format) or NORMAL (a new content question).\n"
            f"Conversation so far:\n{_fmt_history(history)}\n"
            f"User message: {q}\n"
            "Answer with exactly one word: META or NORMAL."
        )
        verdict, cost = _utility_llm(prompt, max_tokens=5)
        route = "meta" if "META" in verdict.upper() else "normal"
        return {"route": route, "utility_cost_usd": state.get("utility_cost_usd", 0.0) + cost}

    return {"route": "normal"}


def resolve_context(state: EvaState) -> dict:
    """
    Resolución de correferencia (MEMORIA). Reescribe follow-ups a preguntas
    autónomas usando el historial. Solo dispara el LLM si (a) hay historial y
    (b) la query trae pronombres colgantes o es muy corta. Si no, pasa directo.
    """
    q = state["question"]
    history = state.get("history", [])

    needs_resolution = bool(history) and (
        _PRONOUN_RE.search(q) is not None or len(q.split()) <= 4
    )
    if not needs_resolution:
        return {"query": q}  # sin costo

    prompt = (
        "Rewrite the user's follow-up as a standalone question, resolving any "
        "pronouns using the conversation. Keep it faithful — do not add info.\n"
        f"Conversation so far:\n{_fmt_history(history)}\n"
        f"Follow-up: {q}\n"
        "Standalone question:"
    )
    resolved, cost = _utility_llm(prompt)
    return {
        "query": resolved or q,
        "utility_cost_usd": state.get("utility_cost_usd", 0.0) + cost,
    }


def retrieve_node(state: EvaState) -> dict:
    """Recupera top-k con el retrieve() existente. Sin cambios al motor."""
    query = state.get("query") or state["question"]
    hits = retrieve(query, k=3)
    best = hits[0]["score"] if hits else 0.0
    return {"retrieved": hits, "best_score": best}


def grade(state: EvaState) -> dict:
    """Decide el branch según el score y los reintentos restantes."""
    score = state.get("best_score", 0.0)
    retries = state.get("retries", 0)

    if score < OFF_TOPIC_THRESHOLD:
        decision = "off_topic"
    elif score < WEAK_BAND_CEIL and retries < MAX_RETRIES:
        decision = "reformulate"
    else:
        # Score bueno, O débil pero ya sin reintentos → respondemos con lo que hay.
        decision = "answer"
    return {"grade": decision}


def reformulate_node(state: EvaState) -> dict:
    """
    Reescribe la query para mejorar el retrieval e incrementa el contador.
    Solo se llega aquí desde la banda débil, y máximo MAX_RETRIES veces.
    """
    query = state.get("query") or state["question"]
    prompt = (
        "The following search query returned weak results against a knowledge "
        "base about a person's CV, skills, and projects. Rewrite it to improve "
        "retrieval — expand abbreviations, add synonyms, be explicit. One line.\n"
        f"Query: {query}\n"
        "Improved query:"
    )
    new_query, cost = _utility_llm(prompt)
    return {
        "query": new_query or query,
        "retries": state.get("retries", 0) + 1,
        "utility_cost_usd": state.get("utility_cost_usd", 0.0) + cost,
    }


def answer_node(state: EvaState) -> dict:
    """Genera la respuesta grounded con el answer() existente."""
    question = state.get("query") or state["question"]
    hits = state.get("retrieved", [])
    text = generate_answer(question, hits)
    sources = [{"source": h["source"], "score": h["score"]} for h in hits]
    return {"answer": text, "sources": sources}


def canned_node(state: EvaState) -> dict:
    """Respuesta enlatada off-topic. Cero tokens de generación."""
    return {"answer": CANNED_OFF_TOPIC, "sources": []}


def meta_node(state: EvaState) -> dict:
    """
    Maneja meta-queries diagnósticas sobre el turno anterior.
      - "sources / fuentes"  → lista las fuentes del último turno del asistente.
      - "más corto / concise"→ re-responde la última pregunta en una sola frase.
    """
    q = state["question"]
    history = state.get("history", [])

    # Última pregunta del usuario y últimas fuentes del asistente en el historial.
    last_user = next((t["content"] for t in reversed(history) if t["role"] == "user"), None)
    last_sources = next(
        (t.get("sources") for t in reversed(history) if t["role"] == "assistant" and t.get("sources")),
        None,
    )

    if _META_CONCISE_RE.search(q) and last_user:
        hits = retrieve(last_user, k=3)
        text = generate_answer(f"{last_user} (answer in one short sentence)", hits)
        sources = [{"source": h["source"], "score": h["score"]} for h in hits]
        return {"answer": text, "sources": sources}

    if last_sources:
        listed = "\n".join(f"  • {s['source']} (score {s['score']:.2f})" for s in last_sources)
        return {"answer": f"That answer drew from:\n{listed}", "sources": last_sources}

    return {
        "answer": "I don't have a previous answer to trace sources for yet. Ask me something first.",
        "sources": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Funciones de ruteo (conditional edges)
# ─────────────────────────────────────────────────────────────────────────────
def _after_route(state: EvaState) -> Literal["meta", "resolve"]:
    return "meta" if state.get("route") == "meta" else "resolve"


def _after_grade(state: EvaState) -> Literal["answer", "reformulate", "off_topic"]:
    return state.get("grade", "answer")  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del grafo
# ─────────────────────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(EvaState)

    # Nota: LangGraph prohíbe que un nombre de nodo colisione con una llave del
    # state. Por eso los nodos que escriben en state["grade"] y state["answer"]
    # se registran con sufijo "_node".
    g.add_node("route_question", route_question)
    g.add_node("resolve_context", resolve_context)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade_node", grade)
    g.add_node("reformulate", reformulate_node)
    g.add_node("answer_node", answer_node)
    g.add_node("canned", canned_node)
    g.add_node("meta", meta_node)

    g.add_edge(START, "route_question")
    g.add_conditional_edges(
        "route_question", _after_route,
        {"meta": "meta", "resolve": "resolve_context"},
    )
    g.add_edge("resolve_context", "retrieve")
    g.add_edge("retrieve", "grade_node")
    g.add_conditional_edges(
        "grade_node", _after_grade,
        {"answer": "answer_node", "reformulate": "reformulate", "off_topic": "canned"},
    )
    g.add_edge("reformulate", "retrieve")  # loop capado por MAX_RETRIES
    g.add_edge("answer_node", END)
    g.add_edge("canned", END)
    g.add_edge("meta", END)

    return g.compile()


_graph = None


def run_turn(question: str, history: list[dict] | None = None) -> EvaState:
    """
    Corre un turno completo. `history` lo mantiene el caller (CLI/Lambda) —
    memoria explícita, stateless-friendly (así se porta a Lambda sin cambios).
    """
    global _graph
    if _graph is None:
        _graph = build_graph()

    initial: EvaState = {
        "question": question,
        "history": history or [],
        "retries": 0,
        "utility_cost_usd": 0.0,
    }
    return _graph.invoke(initial)


if __name__ == "__main__":
    # Smoke test rápido de un solo turno.
    q = " ".join(sys.argv[1:]) or "what are kevin's main skills?"
    result = run_turn(q)
    print(f"Q: {q}")
    print(f"route={result.get('route')} grade={result.get('grade')} "
          f"score={result.get('best_score', 0):.3f} retries={result.get('retries', 0)}")
    print(f"A: {result.get('answer')}")
