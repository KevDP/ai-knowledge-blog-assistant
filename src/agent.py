"""
agent.py - LangGraph orchestration over the EVA RAG core (Phase 2).

Wraps the existing retrieve() + answer() with:
  - Conversation memory (history passed per turn by caller)
  - Hybrid routing (regex heuristic first; LLM fallback in ambiguous cases)
  - Query reformulation when retrieval scores in the weak band
  - Meta-query handling (sources / concise / etc.)
  - Phase 3: two-stage retrieval (bi-encoder recall - cross-encoder rerank)
  - Phase 3: reformulation guard that reverts if the rewritten query degrades

Design principle: this file orchestates. retrieve.py and llm.py remain the engine
nothing here re-implements retrieval or generation. 
The agent decides what to run and in what order.
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
    RELEVANCE_THRESHOLD as OFF_TOPIC_THRESHOLD,  # 0.55 on BGE scale (Phase 0)
    answer as generate_answer,
)
from src.rerank import rerank
from src.retrieve import retrieve

# ─────────────────────────────────────────────────────────────────────────────
# Flow-control constants (and cost guards)
# ─────────────────────────────────────────────────────────────────────────────

# Score bands:
#   score < OFF_TOPIC_THRESHOLD        → off-topic  → canned ($0)
#   OFF_TOPIC_THRESHOLD ≤ score < WEAK → weak       → reformulate (1 retry)
#   score ≥ WEAK                       → strong    → answer
#
# CRITICAL (Phase 3): these thresholds gate the BI-ENCODER score, not the
# rerank score. The bi-encoder answers "is anything remotely relevant in the
# KB?" (recall). The cross-encoder only reorders the already-retrieved
# candidates (precision). Never mix scales between models.
WEAK_BAND_CEIL = 0.62

# Phase 3: two-stage retrieval.
# Wide recall with the bi-encoder (cheap, runs over the whole KB) →
# bounded precision with the cross-encoder (~150ms, runs only over CANDIDATES_K).
CANDIDATES_K = 10   # candidates that pass from the bi-encoder to the reranker
FINAL_K = 3         # final top-k the LLM sees in its context

# Anti-loop guard = anti-cost guard. Without this, reformulate→retrieve→grade
# becomes a cost attack against yourself.
MAX_RETRIES = 1

# Model for "utility" calls (coref, reformulation, routing fallback).
# Haiku because these prompts are short and cheap.
UTILITY_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
UTILITY_MAX_TOKENS = 80

# Meta-query heuristic: words that signal a diagnostic question (about sources)
# or a reformat request (shorter). Regex, $0. Bilingual to match user register.
_META_SOURCES_RE = re.compile(
    r"\b(sources?|fuentes?|cita|citar|de d[oó]nde|where.*from|show.*source)\b",
    re.IGNORECASE,
)
_META_CONCISE_RE = re.compile(
    r"\b(shorter|m[aá]s corto|m[aá]s breve|briefly|concise|res[uú]mel[oa])\b",
    re.IGNORECASE,
)

# Coreference heuristic: dangling pronouns that suggest the question depends
# on the previous turn ("what did HE study?"). Bilingual for user register.
_PRONOUN_RE = re.compile(
    r"\b(he|she|it|they|him|her|that|there|this|"
    r"[ée]l|ella|eso|esa|ese|ah[ií]|ahi|su|sus)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Graph state
# ─────────────────────────────────────────────────────────────────────────────
class EvaState(TypedDict, total=False):
    question: str          # raw user question
    history: list[dict]    # previous turns → MEMORY. [{role, content, sources?}]
    query: str             # resolved/reformulated query used for retrieval
    candidates: list       # top-CANDIDATES_K from the bi-encoder (pre-rerank)
    retrieved: list        # top-FINAL_K post-rerank (what the LLM sees)
    best_score: float      # bi-encoder score of the top candidate (for gating)
    route: str             # "meta" | "normal" (set by route_question)
    grade: str             # "answer" | "reformulate" | "off_topic"
    retries: int           # retry counter (cap = MAX_RETRIES)
    # Phase 3 reformulation guard: pre-reformulation snapshot so we can revert
    # if the rewritten query degrades retrieval instead of improving it.
    pre_reform_score: float
    pre_reform_hits: list
    pre_reform_query: str
    answer: str            # final answer
    sources: list          # sources from the chunks used
    utility_cost_usd: float  # accumulated cost of utility LLM calls this turn


# ─────────────────────────────────────────────────────────────────────────────
# Helper: utility LLM call (coref / reformulation / routing fallback)
# ─────────────────────────────────────────────────────────────────────────────
_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not configured.")
        _client = Anthropic(api_key=api_key)
    return _client


def _utility_llm(prompt: str, *, max_tokens: int = UTILITY_MAX_TOKENS) -> tuple[str, float]:
    """Short, cheap LLM call. Returns (text, estimated_cost_usd)."""
    resp = _get_client().messages.create(
        model=UTILITY_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    u = resp.usage
    # Same placeholder pricing as llm.py
    cost = (u.input_tokens * 1.0 + u.output_tokens * 5.0) / 1_000_000
    return resp.content[0].text.strip(), cost


def _fmt_history(history: list[dict], *, last_n: int = 4) -> str:
    """Serialize the last few turns to give the LLM context."""
    turns = history[-last_n:]
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


# ─────────────────────────────────────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────────────────────────────────────
def route_question(state: EvaState) -> dict:
    """
    HYBRID router. Heuristic first ($0); LLM fallback only when the heuristic
    is uncertain.

    - Meta-query (sources / shorter) detectable by regex → route="meta".
    - Everything else → route="normal".
    - LLM fallback: only when the query is short and ambiguous WITH history
      present (could be an implicit meta). That is the ONLY case that pays
      for routing.
    """
    q = state["question"]
    history = state.get("history", [])

    if _META_SOURCES_RE.search(q) or _META_CONCISE_RE.search(q):
        return {"route": "meta"}

    # Ambiguity: very short query (≤3 words) with history. Could be an
    # implicit meta ("and the sources?") or a normal follow-up. LLM justified.
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
    Coreference resolution (MEMORY). Rewrites follow-ups into standalone
    questions using history. LLM fires only if (a) history exists and (b) the
    query has dangling pronouns or is very short. Otherwise passes through.
    """
    q = state["question"]
    history = state.get("history", [])

    needs_resolution = bool(history) and (
        _PRONOUN_RE.search(q) is not None or len(q.split()) <= 4
    )
    if not needs_resolution:
        return {"query": q}  # zero cost

    prompt = (
        "Rewrite the user's follow-up as a standalone question, resolving any "
        "pronouns using the conversation. Keep it faithful (do not add info).\n"
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
    """
    Retrieval stage 1: bi-encoder recall.

    Fetches the top-CANDIDATES_K candidates via retrieve() (bi-encoder cosine).
    The returned `best_score` feeds the off-topic gate and the weak band.
    those decisions MUST use the bi-encoder scale, not the reranker's.
    """
    query = state.get("query") or state["question"]
    candidates = retrieve(query, k=CANDIDATES_K)
    best_bi_score = candidates[0]["score"] if candidates else 0.0
    return {"candidates": candidates, "best_score": best_bi_score}


def rerank_node(state: EvaState) -> dict:
    """
    Retrieval stage 2: cross-encoder precision.

    Reorders the CANDIDATES_K candidates with a cross-encoder MiniLM and keeps
    the FINAL_K most relevant. Runs locally, no LLM calls, no per-token cost.
    If EVA_RERANK=0, this pass-throughs (useful for A/B evals).

    IMPORTANT: does not touch `best_score`. It stays as the bi-encoder top
    score because downstream gates (off-topic, weak band) reason in BGE scale.
    """
    query = state.get("query") or state["question"]
    candidates = state.get("candidates", [])
    reranked = rerank(query, candidates, top_k=FINAL_K)
    return {"retrieved": reranked}


def grade(state: EvaState) -> dict:
    """
    Picks the branch based on the bi-encoder score and remaining retries.
    Also enforces the REFORMULATION GUARD: if we came from a reformulation and
    the score fell relative to the original, revert query and hits.

    LLM reformulation does not guarantee improvement.
    Sometimes the original query was clearer for the retriever.
    """
    score = state.get("best_score", 0.0)
    retries = state.get("retries", 0)
    pre_score = state.get("pre_reform_score")

    # Phase 3 guard: came from reformulate and retrieval degraded → revert.
    if pre_score is not None and score < pre_score:
        return {
            "retrieved": state.get("pre_reform_hits", state.get("retrieved", [])),
            "best_score": pre_score,
            "query": state.get("pre_reform_query", state.get("query")),
            "grade": "answer",
        }

    if score < OFF_TOPIC_THRESHOLD:
        decision = "off_topic"
    elif score < WEAK_BAND_CEIL and retries < MAX_RETRIES:
        decision = "reformulate"
    else:
        # Strong score, OR weak but out of retries → answer with what we have.
        decision = "answer"
    return {"grade": decision}


def reformulate_node(state: EvaState) -> dict:
    """
    Rewrites the query to improve retrieval and increments the retry counter.
    Only reached from the weak band, at most MAX_RETRIES times.

    Phase 3: saves a pre-reformulation snapshot (query, hits, score) so grade()
    can revert if the rewrite degrades instead of improving.
    """
    query = state.get("query") or state["question"]
    prompt = (
        "The following search query returned weak results against a knowledge "
        "base about a person's CV, skills, and projects. Rewrite it to improve "
        "retrieval (expand abbreviations, add synonyms, be explicit in one line).\n"
        f"Query: {query}\n"
        "Improved query:"
    )
    new_query, cost = _utility_llm(prompt)
    return {
        "query": new_query or query,
        "retries": state.get("retries", 0) + 1,
        "utility_cost_usd": state.get("utility_cost_usd", 0.0) + cost,
        # Snapshot for the reformulation guard in grade().
        "pre_reform_score": state.get("best_score"),
        "pre_reform_hits": state.get("retrieved"),
        "pre_reform_query": query,
    }


def answer_node(state: EvaState) -> dict:
    """Generates the grounded answer via the existing answer() function."""
    question = state.get("query") or state["question"]
    hits = state.get("retrieved", [])
    text = generate_answer(question, hits)
    sources = [{"source": h["source"], "score": h["score"]} for h in hits]
    return {"answer": text, "sources": sources}


def canned_node(state: EvaState) -> dict:
    """Off-topic canned response. Zero generation tokens."""
    return {"answer": CANNED_OFF_TOPIC, "sources": []}


def meta_node(state: EvaState) -> dict:
    """
    Handles diagnostic meta-queries about the previous turn.
      - "sources / fuentes"    → list the sources of the assistant's last turn.
      - "shorter / más corto"  → re-answer the last question in a single sentence.
    """
    q = state["question"]
    history = state.get("history", [])

    # Last user question and last assistant sources in history.
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
# Routing functions (conditional edges)
# ─────────────────────────────────────────────────────────────────────────────
def _after_route(state: EvaState) -> Literal["meta", "resolve"]:
    return "meta" if state.get("route") == "meta" else "resolve"


def _after_grade(state: EvaState) -> Literal["answer", "reformulate", "off_topic"]:
    return state.get("grade", "answer")  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(EvaState)

    # Note: LangGraph forbids node names that collide with state keys. That is
    # why nodes writing to state["grade"] and state["answer"] are registered
    # with a "_node" suffix.
    g.add_node("route_question", route_question)
    g.add_node("resolve_context", resolve_context)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank_node", rerank_node)  # Phase 3: precision stage
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
    g.add_edge("retrieve", "rerank_node")   # Phase 3: two-stage retrieval
    g.add_edge("rerank_node", "grade_node")
    g.add_conditional_edges(
        "grade_node", _after_grade,
        {"answer": "answer_node", "reformulate": "reformulate", "off_topic": "canned"},
    )
    g.add_edge("reformulate", "retrieve")  # loop capped by MAX_RETRIES
    g.add_edge("answer_node", END)
    g.add_edge("canned", END)
    g.add_edge("meta", END)

    return g.compile()


_graph = None


def run_turn(question: str, history: list[dict] | None = None) -> EvaState:
    """
    Runs a full turn. `history` is maintained by the caller (CLI/Lambda) -
    explicit memory, stateless-friendly (portable to Lambda without changes).
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
    # Quick single-turn smoke test.
    q = " ".join(sys.argv[1:]) or "what are kevin's main skills?"
    result = run_turn(q)
    print(f"Q: {q}")
    print(f"route={result.get('route')} grade={result.get('grade')} "
          f"score={result.get('best_score', 0):.3f} retries={result.get('retries', 0)}")
    print(f"A: {result.get('answer')}")
