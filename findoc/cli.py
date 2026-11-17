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
    """Write the annotated original, flat page, enhanced page and registered cheque."""
    from findoc.config import get_settings
    from findoc.localize.templates import default_store, draw_fields
    from findoc.preprocess.detect import draw_detection

    written: list[Path] = []
    stem = doc.source.stem
    if doc.frames:  # video input
        from findoc.track.viz import keyframe_sheet, write_tracking_video

        written.append(write_tracking_video(doc, save_dir / f"{stem}_tracking.mp4"))
        if doc.keyframes:
            written.append(save_dir / f"{stem}_keyframes.jpg")
            save_image(written[-1], keyframe_sheet(doc))
        if doc.original is not None:
            written.append(save_dir / f"{stem}_best.jpg")
            save_image(written[-1], doc.original)
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
    if doc.registered is not None and doc.template_id:
        template = default_store(get_settings().templates_dir).get(doc.template_id)
        written.append(save_dir / f"{stem}_fields.png")
        save_image(written[-1], draw_fields(doc.registered, template))
    return written


@app.command("process")
def process_cmd(
    source: Path = typer.Argument(..., exists=True, dir_okay=False, help="Image or video file"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write result JSON here"),
    save_dir: Path | None = typer.Option(
        None,
        "--save-dir",
        "-s",
        help="Save detected/flat/enhanced images (video: tracking video, keyframes) here",
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

    if doc.tracking:
        console.print(
            "Tracking: " + ", ".join(f"{k}={v}" for k, v in doc.tracking.items()), highlight=False
        )
        console.print(f"Keyframes: {[k.frame for k in doc.keyframes]}", highlight=False)
    if doc.quality:
        console.print(
            "Quality: " + ", ".join(f"{k}={v}" for k, v in doc.quality.items()), highlight=False
        )
    colour = {"ok": "green", "rejected_quality": "yellow"}.get(doc.status.value, "red")
    console.print(f"Status: [bold {colour}]{doc.status.value}[/]")
    for msg in doc.messages:
        console.print(f"  - {msg}")
    if doc.fields:
        ft = Table(title=f"Cheque fields ({doc.template_id})")
        for col in ("Field", "Read", "Validated", "Conf"):
            ft.add_column(col)
        for f in doc.fields:
            ok = f.normalized is not None
            ft.add_row(
                f.name, f.value, f.normalized if ok else "[red]invalid[/]", f"{f.confidence:.2f}"
            )
        console.print(ft)
    if doc.checks:
        ct = Table(title="Checks")
        for col in ("Check", "Result", "Reason"):
            ct.add_column(col)
        for c in doc.checks:
            ct.add_row(c.id, "[red]FLAG[/]" if c.fired else "[green]pass[/]", c.reason)
        console.print(ct)
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
    n: int = typer.Option(10, "--n", help="Number of photos or videos"),
    out_dir: Path | None = typer.Option(None, "--out", "-o", help="Default: data/synthetic/<kind>"),
    kind: str = typer.Option("page", help="'page' (generic document), 'cheque' or 'video'"),
    seed: int = typer.Option(0, help="Random seed (same seed = same photos)"),
    blur: float = typer.Option(0.0, help="Gaussian blur sigma, to simulate camera shake"),
    fraud_rate: float = typer.Option(
        0.4, help="Cheques only: share with a fraud condition (mismatch/stale/postdated/unsigned)"
    ),
    seconds: float = typer.Option(4.0, help="Videos only: clip length"),
) -> None:
    """Generate synthetic photos with ground truth (labels.json)."""
    out_dir = out_dir or Path("data/synthetic") / kind
    if kind == "page":
        from findoc.synth.photo import generate

        paths = generate(n, out_dir, seed=seed, blur_sigma=blur)
    elif kind == "cheque":
        from findoc.synth.dataset import generate_cheques

        paths = generate_cheques(n, out_dir, seed=seed, blur_sigma=blur, fraud_rate=fraud_rate)
    elif kind == "video":
        from findoc.synth.video import generate_videos

        paths = generate_videos(n, out_dir, seed=seed, fraud_rate=fraud_rate, seconds=seconds)
    else:
        raise typer.BadParameter("kind must be 'page', 'cheque' or 'video'")
    what = "videos" if kind == "video" else "photos"
    console.print(f"Wrote {len(paths)} {what} and labels.json to {out_dir}")


templates_app = typer.Typer(help="Manage cheque templates.", no_args_is_help=True)
app.add_typer(templates_app, name="templates")


@templates_app.command("list")
def templates_list() -> None:
    """Show the templates in use."""
    from findoc.localize.templates import default_store

    store = default_store(get_settings().templates_dir)
    for t in store:
        console.print(f"{t.id}  {t.name}  {t.size[0]}x{t.size[1]}  fields={len(t.fields)}")


@templates_app.command("export")
def templates_export(out_dir: Path = typer.Option(Path("data/templates"), "--out", "-o")) -> None:
    """Write the built-in templates to a folder (edit them or add your own, then set
    FINDOC_TEMPLATES_DIR to that folder)."""
    from findoc.localize.templates import builtin_store

    for path in builtin_store().save(out_dir):
        console.print(f"Wrote {path}")


if __name__ == "__main__":
    app()
