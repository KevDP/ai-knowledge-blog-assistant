"""
chat.py

CLI interactivo para EVA. Fase 2: ahora corre sobre el grafo LangGraph
(src.agent), no sobre retrieve()+answer() directo.

Novedades vs Fase 0:
- Mantiene el HISTORIAL de la conversación entre turnos (memoria). El grafo
  es stateless; el estado vive aquí, en el REPL. Así se porta a Lambda sin
  cambios: allá el historial vendría del request o de DynamoDB.
- Muestra la decisión de routing (route/grade/score/retries/cost) en un panel
  tenue, para que se vea el comportamiento "agéntico".

Uso:
    python -m src.chat
    python -m src.chat --once "what is kevin's experience?"
    python -m src.chat --show-trace       # muestra el routing en cada turno
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent import run_turn

console = Console()

# El caller acota el input igual que en Fase 0. El grafo asume que la pregunta
# ya pasó este gate barato antes de gastar en embeddings o LLM.
MAX_QUESTION_CHARS = 500


def _trace_line(result: dict) -> str:
    """Resumen de una línea de la decisión del grafo para este turno."""
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
    Corre un turno, imprime la respuesta y devuelve el historial ACTUALIZADO.
    El historial se pasa por valor y se regresa extendido — el REPL lo reusa
    en el siguiente turno para dar memoria.
    """
    if len(question) > MAX_QUESTION_CHARS:
        console.print(
            f"[yellow]Pregunta muy larga ({len(question)} chars). "
            f"Límite: {MAX_QUESTION_CHARS}.[/yellow]"
        )
        return history

    with console.status("[dim]pensando (routing → retrieve → answer)..."):
        result = run_turn(question, history=history)

    answer_text = result.get("answer", "")
    sources = result.get("sources", [])

    console.print(Panel(Markdown(answer_text), title="EVA", border_style="cyan"))

    if show_trace:
        console.print(f"[dim]{_trace_line(result)}[/dim]")
    if show_sources and sources:
        console.print("[dim]Sources:[/dim]")
        console.print(f"[dim]{_format_sources(sources)}[/dim]")

    # Extiende el historial: turno del usuario + turno del asistente (con fuentes,
    # para que las meta-queries "show sources" puedan trazarlas después).
    new_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer_text, "sources": sources},
    ]
    return new_history


def repl(*, show_sources: bool, show_trace: bool) -> None:
    console.print(Panel(
        "EVA — RAG agent demo (Fase 2)\n"
        "Con memoria de conversación. Prueba un follow-up:\n"
        "  'who is kevin?'  →  'what did he study?'\n"
        "Escribe 'exit' o Ctrl-C para salir.",
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
    load_dotenv()  # lee .env del cwd

    parser = argparse.ArgumentParser(description="EVA RAG agent CLI (Fase 2)")
    parser.add_argument("--once", type=str, default=None, help="Pregunta única y sale")
    parser.add_argument("--show-sources", action="store_true", help="Imprime las fuentes usadas")
    parser.add_argument("--show-trace", action="store_true", help="Imprime la decisión de routing")
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
