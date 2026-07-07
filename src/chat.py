"""
chat.py

Interactive CLI for EVA. Phase 2: runs over the LangGraph pipeline
(src.agent) rather than calling retrieve() + answer() directly.

What is new vs Phase 0:
- Maintains conversation HISTORY across turns (memory). The graph is
  stateless; state lives here in the REPL. That way it ports to Lambda
  without changes (history would come from the request or DynamoDB there).
- Shows the routing decision (route/grade/score/retries/cost) in a dim
  trace panel so the "agentic" behavior is visible when demoing.

Usage:
    python -m src.chat
    python -m src.chat --once "what is kevin's experience?"
    python -m src.chat --show-trace       # show routing on every turn
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent import run_turn

console = Console()

# Caller-side length cap, mirroring Phase 0. The graph assumes the question
# has already passed this cheap gate before spending on embeddings or LLM.
MAX_QUESTION_CHARS = 500


def _trace_line(result: dict) -> str:
    """One-line summary of the graph's decision for this turn."""
    return (
        f"route={result.get('route', '—')}  "
        f"grade={result.get('grade', '—')}  "
        f"score={result.get('best_score', 0.0):.3f}  "
        f"retries={result.get('retries', 0)}  "
        f"util_cost=${result.get('utility_cost_usd', 0.0):.6f}"
    )


def _format_sources(sources) -> str:
    return "\n".join(
        f"  • {s['source']} (score {s['score']:.2f})" for s in sources
    )


def run_and_print(
    question: str,
    history: list[dict],
    *,
    show_sources: bool = False,
    show_trace: bool = False,
) -> list[dict]:
    """
    Runs one turn, prints the answer, and returns the UPDATED history.
    History is passed by value and returned extended.
    The REPL reuses it on the next turn to provide memory.
    """
    if len(question) > MAX_QUESTION_CHARS:
        console.print(
            f"[yellow]Question too long ({len(question)} chars). "
            f"Limit: {MAX_QUESTION_CHARS}.[/yellow]"
        )
        return history

    with console.status("[dim]thinking (routing → retrieve → answer)..."):
        result = run_turn(question, history=history)

    answer_text = result.get("answer", "")
    sources = result.get("sources", [])

    console.print(Panel(Markdown(answer_text), title="EVA", border_style="cyan"))

    if show_trace:
        console.print(f"[dim]{_trace_line(result)}[/dim]")
    if show_sources and sources:
        console.print("[dim]Sources:[/dim]")
        console.print(f"[dim]{_format_sources(sources)}[/dim]")

    # Extend history: user turn + assistant turn (with sources so later
    # meta-queries like "show sources" can trace them).
    new_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer_text, "sources": sources},
    ]
    return new_history


def repl(*, show_sources: bool, show_trace: bool) -> None:
    console.print(Panel(
        "EVA - RAG agent demo (Phase 2)\n"
        "With conversation memory. Try a follow-up:\n"
        "  'who is kevin?' + 'what did he study?'\n"
        "Type 'exit' or Ctrl-C to quit.",
        border_style="cyan",
    ))
    history: list[dict] = []
    while True:
        try:
            question = console.input("[bold cyan]you ›[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye.[/dim]")
            return
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            console.print("[dim]bye.[/dim]")
            return
        try:
            history = run_and_print(
                question, history,
                show_sources=show_sources, show_trace=show_trace,
            )
        except Exception as exc:
            console.print(f"[red]error:[/red] {exc}")


def main() -> None:
    load_dotenv()  # loads .env from the current working directory

    parser = argparse.ArgumentParser(description="EVA RAG agent CLI (Phase 2+)")
    parser.add_argument("--once", type=str, default=None, help="Single question, then exit")
    parser.add_argument("--show-sources", action="store_true", help="Print the sources used")
    parser.add_argument("--show-trace", action="store_true", help="Print the routing decision")
    args = parser.parse_args()

    if args.once:
        run_and_print(
            args.once, [],
            show_sources=args.show_sources, show_trace=args.show_trace,
        )
    else:
        repl(show_sources=args.show_sources, show_trace=args.show_trace)


if __name__ == "__main__":
    main()
