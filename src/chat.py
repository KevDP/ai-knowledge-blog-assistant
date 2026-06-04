"""
chat.py — CLI interactivo para Fase 0.

Es el orquestador: lee pregunta → retrieve → llm.answer → imprime.
No contiene lógica de negocio; solo glue + UX de terminal.

Uso:
    python -m src.chat
    python -m src.chat --once "what is kevin's experience?"
    python -m src.chat --show-sources
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.llm import answer
from src.retrieve import retrieve

console = Console()

# Defensa de costo: cualquier pregunta más larga que esto se rechaza antes
# de embebir o llamar al LLM. 500 chars ≈ ~120 tokens — más que suficiente
# para una pregunta legítima sobre el portafolio. Tunear si limita demasiado.
MAX_QUESTION_CHARS = 500


def _format_sources(hits) -> str:
    lines = [
        f"  • {hit['source']} (score {hit['score']:.2f})"
        for hit in hits
    ]
    return "\n".join(lines)


def ask_once(question: str, *, k: int = 3, show_sources: bool = False) -> None:
    if len(question) > MAX_QUESTION_CHARS:
        console.print(
            f"[yellow]Question too long ({len(question)} chars). "
            f"Limit: {MAX_QUESTION_CHARS}. Be more concise.[/yellow]"
        )
        return

    with console.status("[dim]retrieving..."):
        hits = retrieve(question, k=k)

    with console.status("[dim]thinking..."):
        text = answer(question, hits)

    console.print(Panel(Markdown(text), title="EVA", border_style="cyan"))
    if show_sources and hits:
        console.print("[dim]Sources:[/dim]")
        console.print(f"[dim]{_format_sources(hits)}[/dim]")


def repl(*, k: int, show_sources: bool) -> None:
    console.print(Panel(
        "EVA — local RAG demo (Phase 0)\n"
        "Ask me anything about Kevin. Type 'exit' or Ctrl-C to quit.",
        border_style="cyan",
    ))
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
            ask_once(question, k=k, show_sources=show_sources)
        except Exception as exc:  # noqa: BLE001 — UX: mostramos cualquier error y seguimos.
            console.print(f"[red]error:[/red] {exc}")


def main() -> None:
    load_dotenv()  # lee .env del cwd

    parser = argparse.ArgumentParser(description="EVA local RAG CLI (Phase 0)")
    parser.add_argument("--once", type=str, default=None, help="Pregunta única y sale")
    parser.add_argument("-k", type=int, default=3, help="Top-k chunks a recuperar")
    parser.add_argument("--show-sources", action="store_true", help="Imprime las fuentes usadas")
    args = parser.parse_args()

    if args.once:
        ask_once(args.once, k=args.k, show_sources=args.show_sources)
    else:
        repl(k=args.k, show_sources=args.show_sources)


if __name__ == "__main__":
    main()
