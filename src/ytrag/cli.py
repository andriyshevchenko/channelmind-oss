from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from .chunk import chunk_transcript
from .config import Config, load_config
from .embed import make_embedder
from .ingest import download_transcripts, load_transcripts
from .rag import Assistant, RETRIEVAL_TOP_K
from .store import VectorStore
from .throttle import ThrottleConfig
from .usage import UsageStore

app = typer.Typer(help="Download a YouTube channel's transcripts and chat with them.")
console = Console()

_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "voyage": "VOYAGE_API_KEY",
}


def _require(value: str, name: str) -> None:
    if not value:
        console.print(f"[red]Missing {name}. Set it in your .env file.[/red]")
        raise typer.Exit(1)


def _require_embed_key(cfg: Config) -> None:
    _require(cfg.embed_key(), _KEY_ENV[cfg.embed_provider])


def _require_llm_key(cfg: Config) -> None:
    _require(cfg.llm_key(), _KEY_ENV[cfg.llm_provider])


@app.command()
def ingest(
    channel: str = typer.Argument(..., help="Channel handle (@name), name, or URL."),
    langs: str = typer.Option("", help="Comma-separated subtitle languages. Blank = the video's original source language; use 'all' for every language the channel offers."),
    limit: int = typer.Option(None, help="Only fetch the N most recent videos."),
    max_retries: int = typer.Option(0, help="Retries per blocked video. 0 = patient/unlimited (never fail)."),
    cookies_browser: str = typer.Option("", help="Read logged-in cookies from this browser (chrome, edge, firefox...) for age-gated videos."),
    base_delay: float = typer.Option(2.0, help="Base seconds between requests (paced)."),
    max_delay: float = typer.Option(900.0, help="Cap on paced delay after backoff."),
):
    """Download transcripts for every video on a channel (patient & resumable)."""
    cfg = load_config()
    langs_list = [x.strip() for x in langs.split(",") if x.strip()]
    retries = None if max_retries <= 0 else max_retries
    throttle = ThrottleConfig(base_delay=base_delay, max_delay=max_delay)
    console.print(f"[cyan]Fetching transcripts for {channel}...[/cyan]")
    results = download_transcripts(
        channel,
        cfg.transcripts_dir,
        langs=langs_list,
        limit=limit,
        throttle=throttle,
        max_retries=retries,
        cookies_browser=cookies_browser or None,
        progress=lambda ev: console.print(f"[dim]{ev.get('message', '')}[/dim]"),
        proxy=cfg.transcript_proxy or None,
    )
    console.print(f"[green]Saved/available {len(results)} transcripts in {cfg.transcripts_dir}[/green]")


@app.command()
def search(query: str = typer.Argument(...), top_k: int = typer.Option(RETRIEVAL_TOP_K)):
    """Retrieval only: print the transcript excerpts most relevant to a query.

    Handy for active AI sessions — run this in the shell and feed the output back
    into the model, no MCP or pre-registration needed."""
    cfg = load_config()
    _require_embed_key(cfg)
    from .embed import make_embedder

    embedder = make_embedder(cfg)
    store = VectorStore(cfg.chroma_dir)
    hits = store.query(embedder.embed_query(query), top_k=top_k)
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        console.print(f"[bold]{i}. {m['title']}[/bold] ({m['url']})")
        console.print(h["text"])
        console.print("")


@app.command()
def build(
    chunk_words: int = typer.Option(350, help="Words per chunk."),
    overlap: int = typer.Option(60, help="Overlap words between chunks."),
):
    """Chunk downloaded transcripts, embed them, and build the vector index."""
    cfg = load_config()
    _require_embed_key(cfg)
    docs = load_transcripts(cfg.transcripts_dir)
    if not docs:
        console.print("[yellow]No transcripts found. Run `ytrag ingest` first.[/yellow]")
        raise typer.Exit(1)

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_transcript(doc, chunk_words, overlap))
    console.print(f"[cyan]Embedding {len(all_chunks)} chunks from {len(docs)} videos...[/cyan]")

    embedder = make_embedder(cfg)
    embeddings = embedder.embed_documents([c.text for c in all_chunks])
    store = VectorStore(cfg.chroma_dir)
    store.add(all_chunks, embeddings)
    console.print(f"[green]Index ready: {store.count()} chunks in {cfg.chroma_dir}[/green]")


@app.command()
def ask(question: str = typer.Argument(...), top_k: int = typer.Option(RETRIEVAL_TOP_K)):
    """Ask a single question against the indexed transcripts."""
    cfg = load_config()
    _require_llm_key(cfg)
    _require_embed_key(cfg)
    result = Assistant(cfg).answer(question, top_k=top_k)
    console.print(Markdown(result["answer"]))
    _print_sources(result["sources"])


@app.command()
def chat(top_k: int = typer.Option(RETRIEVAL_TOP_K)):
    """Start an interactive chat grounded in the transcripts."""
    cfg = load_config()
    _require_llm_key(cfg)
    _require_embed_key(cfg)
    assistant = Assistant(cfg)
    history: list[dict] = []
    console.print("[cyan]Chat started. Type 'exit' to quit.[/cyan]")
    while True:
        try:
            q = console.input("[bold green]you>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        result = assistant.answer(q, history=history, top_k=top_k)
        console.print(Markdown(result["answer"]))
        _print_sources(result["sources"])
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": result["answer"]})


@app.command()
def web(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8000)):
    """Launch the local web UI (ingest + settings + test chat)."""
    from .web.app import run

    console.print(f"[cyan]Web UI on http://{host}:{port}[/cyan]")
    run(host=host, port=port)


@app.command()
def serve(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8010)):
    """Run the agent-facing MCP server (/mcp) + admin plane (/admin).

    Set YTRAG_ADMIN_KEY to enable the admin plane. Agents connect to /mcp with an
    `Authorization: Bearer <corpus-token>` header minted by the admin plane."""
    import uvicorn

    from .logging_setup import setup_logging
    from .server import build_server

    setup_logging()  # stdout app logs (respects YTRAG_LOG_LEVEL) for the MCP plane
    console.print(f"[cyan]MCP + admin server on http://{host}:{port}[/cyan]")
    console.print(f"[dim]  MCP   : http://{host}:{port}/mcp[/dim]")
    console.print(f"[dim]  Admin : http://{host}:{port}/admin[/dim]")
    uvicorn.run(build_server(), host=host, port=port)


@app.command()
def info():
    """Show configuration and index status."""
    cfg = load_config()
    store = VectorStore(cfg.chroma_dir)
    console.print(f"LLM     : {cfg.llm_provider} / {cfg.llm_model}")
    console.print(f"Embed   : {cfg.embed_provider} / {cfg.embed_model}")
    console.print(f"Transcripts   : {cfg.transcripts_dir}")
    console.print(f"Indexed chunks: {store.count()}")


@app.command()
def usage():
    """Show recorded token usage and estimated spend, per bot owner."""
    cfg = load_config()
    summary = UsageStore(cfg.data_dir).summary()
    users = summary["users"]
    if not users:
        console.print("[yellow]No usage recorded yet.[/yellow]")
        return

    table = Table(title="Token usage & estimated cost")
    table.add_column("user_id", style="cyan")
    table.add_column("calls", justify="right")
    table.add_column("prompt_tokens", justify="right")
    table.add_column("completion_tokens", justify="right")
    table.add_column("cost_usd", justify="right", style="green")
    for user_id, rec in users.items():
        table.add_row(
            user_id,
            str(rec.get("calls", 0)),
            str(rec.get("prompt_tokens", 0)),
            str(rec.get("completion_tokens", 0)),
            f"{rec.get('total_cost_usd', 0.0):.6f}",
        )
    console.print(table)

    g = summary["grand_total"]
    console.print(
        f"[bold]Grand total[/bold]: {g['calls']} calls, "
        f"{g['prompt_tokens']} prompt + {g['completion_tokens']} completion tokens, "
        f"[green]${g['total_cost_usd']:.6f}[/green]"
    )


def _print_sources(hits: list[dict]) -> None:
    if not hits:
        return
    seen = []
    for h in hits:
        title = h["meta"]["title"]
        if title not in seen:
            seen.append(title)
    console.print("\n[dim]Sources: " + "; ".join(seen) + "[/dim]")


if __name__ == "__main__":
    app()
