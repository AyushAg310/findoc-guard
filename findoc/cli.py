"""Command-line interface: ``findoc --help``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from findoc import __version__
from findoc.config import get_settings
from findoc.io import save_image
from findoc.logging import setup_logging
from findoc.pipeline import process
from findoc.types import Document

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


def save_artifacts(doc: Document, save_dir: Path) -> list[Path]:
    """Write the annotated original, flat page and enhanced page as images."""
    from findoc.preprocess.detect import draw_detection

    written: list[Path] = []
    stem = doc.source.stem
    if doc.original is not None and doc.corners is not None:
        overlay = draw_detection(doc.original, np.array(doc.corners, dtype=np.float32))
        written.append(save_dir / f"{stem}_detected.jpg")
        save_image(written[-1], overlay)
    if doc.flat is not None:
        written.append(save_dir / f"{stem}_flat.png")
        save_image(written[-1], doc.flat)
    if doc.enhanced is not None:
        written.append(save_dir / f"{stem}_enhanced.png")
        save_image(written[-1], doc.enhanced)
    return written


@app.command("process")
def process_cmd(
    source: Path = typer.Argument(..., exists=True, dir_okay=False, help="Image or video file"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write result JSON here"),
    save_dir: Path | None = typer.Option(
        None, "--save-dir", "-s", help="Save detected/flat/enhanced images in this folder"
    ),
    ocr: bool = typer.Option(True, "--ocr/--no-ocr", help="Run OCR (needs the [ocr] extra)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Run the pipeline on one photo or video."""
    settings = get_settings()
    setup_logging("DEBUG" if verbose else settings.log_level)

    doc = process(source, ocr=ocr)
    result = doc.result()

    table = Table(title=f"FinDoc Guard - {source.name}")
    table.add_column("Stage")
    table.add_column("ms", justify="right")
    table.add_column("OK")
    for s in doc.stages:
        table.add_row(s.stage, f"{s.ms:.1f}", "yes" if s.ok else f"no: {s.note}")
    console.print(table)

    if doc.quality:
        console.print(
            "Quality: " + ", ".join(f"{k}={v}" for k, v in doc.quality.items()), highlight=False
        )
    colour = {"ok": "green", "rejected_quality": "yellow"}.get(doc.status.value, "red")
    console.print(f"Status: [bold {colour}]{doc.status.value}[/]")
    for msg in doc.messages:
        console.print(f"  - {msg}")
    if doc.full_text:
        console.print("[bold]Text:[/]")
        console.print(doc.full_text, highlight=False)

    if save_dir:
        for path in save_artifacts(doc, save_dir):
            console.print(f"Saved {path}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        console.print(f"Result written to {out}")

    if doc.status.value == "failed":
        raise typer.Exit(code=1)


@app.command()
def synth(
    n: int = typer.Option(10, "--n", help="Number of photos"),
    out_dir: Path = typer.Option(Path("data/synthetic/photos"), "--out", "-o"),
    seed: int = typer.Option(0, help="Random seed (same seed = same photos)"),
    blur: float = typer.Option(0.0, help="Gaussian blur sigma, to simulate camera shake"),
) -> None:
    """Generate synthetic document photos with ground-truth corners."""
    from findoc.synth.photo import generate

    paths = generate(n, out_dir, seed=seed, blur_sigma=blur)
    console.print(f"Wrote {len(paths)} photos and labels.json to {out_dir}")


if __name__ == "__main__":
    app()
