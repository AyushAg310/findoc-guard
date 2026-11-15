"""Command-line interface: ``findoc --help``."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from findoc import __version__
from findoc.config import get_settings
from findoc.logging import setup_logging
from findoc.pipeline import process

app = typer.Typer(
    name="findoc",
    help="FinDoc Guard: 3D-aware cheque and invoice processing with fraud detection.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"findoc-guard {__version__}")


@app.command("process")
def process_cmd(
    source: Path = typer.Argument(..., exists=True, dir_okay=False, help="Image or video file"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write result JSON here"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Run the pipeline on one photo or video."""
    settings = get_settings()
    setup_logging("DEBUG" if verbose else settings.log_level)

    doc = process(source)
    result = doc.result()

    table = Table(title=f"FinDoc Guard - {source.name}")
    table.add_column("Stage")
    table.add_column("ms", justify="right")
    table.add_column("OK")
    for s in doc.stages:
        table.add_row(s.stage, f"{s.ms:.1f}", "yes" if s.ok else f"no: {s.note}")
    console.print(table)
    console.print(f"Status: [bold]{doc.status.value}[/bold]")
    for msg in doc.messages:
        console.print(f"  - {msg}")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        console.print(f"Result written to {out}")

    if doc.status.value == "failed":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
