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
        if doc.recon is not None:
            from findoc.recon3d.export import save_recon

            written += save_recon(doc.recon, save_dir, stem)
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
    if doc.tamper_map is not None and doc.flat is not None:
        from findoc.fraud.viz import tamper_overlay

        written.append(save_dir / f"{stem}_tamper.jpg")
        save_image(written[-1], tamper_overlay(doc.flat, doc.tamper_map))
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
    register: bool = typer.Option(
        False, "--register", help="Add this cheque to the duplicate index after checking it"
    ),
    account: str | None = typer.Option(
        None, "--account", help="Account whose reference signatures to compare with"
    ),
) -> None:
    """Run the pipeline on one photo or video."""
    settings = get_settings()
    setup_logging("DEBUG" if verbose else settings.log_level)

    doc = process(source, ocr=ocr, register=register, account=account)
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
    if doc.recon3d:
        console.print(
            "3D: " + ", ".join(f"{k}={v}" for k, v in doc.recon3d.items()), highlight=False
        )
    if doc.quality:
        console.print(
            "Quality: " + ", ".join(f"{k}={v}" for k, v in doc.quality.items()), highlight=False
        )
    colour = {"ok": "green", "rejected_quality": "yellow"}.get(doc.status.value, "red")
    console.print(f"Status: [bold {colour}]{doc.status.value}[/]")
    for msg in doc.messages:
        console.print(f"  - {msg}")
    if doc.fields:
        title = (
            f"Cheque fields ({doc.template_id})"
            if doc.template_id
            else f"{doc.doc_type.value.title()} fields"
        )
        ft = Table(title=title)
        for col in ("Field", "Read", "Validated", "Conf"):
            ft.add_column(col)
        for f in doc.fields:
            ok = f.normalized is not None
            ft.add_row(
                f.name, f.value, f.normalized if ok else "[red]invalid[/]", f"{f.confidence:.2f}"
            )
        console.print(ft)
    if doc.line_items:
        lt = Table(title="Line items")
        for col in ("#", "Description", "HSN", "Qty", "Rate", "Amount"):
            lt.add_column(col)
        for it in doc.line_items:
            n = it.get("normalized", {})
            lt.add_row(
                str(it["row"]),
                it.get("desc", ""),
                n.get("hsn") or "",
                n.get("qty") or "",
                n.get("rate") or "",
                n.get("amount") or "",
            )
        console.print(lt)
    if doc.checks:
        ct = Table(title="Checks")
        for col in ("Check", "Result", "Reason"):
            ct.add_column(col)
        for c in doc.checks:
            ct.add_row(c.id, "[red]FLAG[/]" if c.fired else "[green]pass[/]", c.reason)
        console.print(ct)
    if doc.risk:
        band = doc.risk["band"]
        colour = {"low": "green", "medium": "yellow", "high": "red"}[band]
        console.print(f"Risk score: [bold {colour}]{doc.risk['score']}/100 ({band})[/]")
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
    kind: str = typer.Option(
        "page", help="'page' (generic document), 'cheque', 'video' or 'invoice' (GST)"
    ),
    seed: int = typer.Option(0, help="Random seed (same seed = same photos)"),
    blur: float = typer.Option(0.0, help="Gaussian blur sigma, to simulate camera shake"),
    fraud_rate: float = typer.Option(
        0.4, help="Cheques only: share with a fraud condition (mismatch/stale/postdated/unsigned)"
    ),
    seconds: float = typer.Option(4.0, help="Videos only: clip length"),
    bend: str = typer.Option(
        "flat", help="Videos only: page shape: flat, wave, curl, curl2 (both ends) or fold"
    ),
) -> None:
    """Generate synthetic photos with ground truth (labels.json)."""
    out_dir = out_dir or Path("data/synthetic") / kind
    if kind == "page":
        from findoc.synth.photo import generate

        paths = generate(n, out_dir, seed=seed, blur_sigma=blur)
    elif kind == "cheque":
        from findoc.synth.dataset import generate_cheques

        paths = generate_cheques(n, out_dir, seed=seed, blur_sigma=blur, fraud_rate=fraud_rate)
    elif kind == "invoice":
        from findoc.synth.invoice import generate_invoices

        paths = generate_invoices(n, out_dir, seed=seed, fraud_rate=fraud_rate, blur_sigma=blur)
    elif kind == "video":
        from findoc.synth.surface import BEND_KINDS
        from findoc.synth.video import generate_videos

        if bend not in BEND_KINDS:
            raise typer.BadParameter(f"bend must be one of {', '.join(BEND_KINDS)}")
        paths = generate_videos(
            n, out_dir, seed=seed, fraud_rate=fraud_rate, seconds=seconds, bend=bend
        )
    else:
        raise typer.BadParameter("kind must be 'page', 'cheque', 'video' or 'invoice'")
    what = "videos" if kind == "video" else "photos"
    console.print(f"Wrote {len(paths)} {what} and labels.json to {out_dir}")


segment_app = typer.Typer(help="Learned page detection (Phase 4).", no_args_is_help=True)
app.add_typer(segment_app, name="segment")


@segment_app.command("train")
def segment_train(
    out: Path = typer.Option(Path("models/docseg.onnx"), "--out", "-o", help="ONNX model to write"),
    samples: int = typer.Option(20000, help="Synthetic training images per epoch"),
    epochs: int = typer.Option(12),
    batch_size: int = typer.Option(16, "--batch-size"),
    workers: int = typer.Option(
        -1, help="Data-generation worker processes (-1 = CPU cores minus 2, at most 8)"
    ),
    device: str = typer.Option("auto", help="'auto', 'cuda' or 'cpu'"),
) -> None:
    """Train the document segmenter on synthetic data and export it to ONNX (needs PyTorch)."""
    try:
        from findoc.segment.train import train
    except ImportError as exc:
        raise typer.BadParameter(
            f'Training needs PyTorch ({exc}). Install it, e.g. pip install -e ".[ocr]"'
        ) from exc
    result = train(
        out,
        samples=samples,
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        device=device,
        log=console.print,
    )
    console.print(
        f"Best validation IoU {result.best_val_iou:.4f} on {result.device} in "
        f"{result.seconds / 60:.1f} min; model written to {result.onnx_path}"
    )


@segment_app.command("check")
def segment_check(
    image: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(Path("outputs/segment_check.jpg"), "--out", "-o"),
) -> None:
    """Show what the segmenter sees on one image: probability mask and the fitted outline."""
    import cv2

    from findoc.io import load_image
    from findoc.segment.infer import load_segmenter

    model = load_segmenter(str(get_settings().segmenter_model))
    if model is None:
        raise typer.BadParameter(f"No model at {get_settings().segmenter_model}; train one first")
    img = load_image(image)
    seg = model.segment(img)
    heat = cv2.applyColorMap((seg.mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    view = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
    if seg.corners is not None:
        cv2.polylines(view, [seg.corners.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 3)
    save_image(out, view)
    console.print(f"agreement {seg.agreement:.3f}; saved {out}")


@app.command()
def calibrate(
    source: Path = typer.Argument(..., exists=True, help="Video or folder of checkerboard photos"),
    board: str = typer.Option("9x6", help="Inner corners per row x per column"),
    step: int = typer.Option(10, help="Use every n-th video frame"),
    save: bool = typer.Option(False, "--save", help="Write FINDOC_FOCAL_PX to .env"),
) -> None:
    """Measure the camera's focal length from a printed checkerboard (improves 3D)."""
    from findoc.recon3d.calibrate import calibrate as run_calibration
    from findoc.recon3d.calibrate import load_images, save_focal

    cols, rows = (int(x) for x in board.lower().split("x"))
    try:
        cal = run_calibration(load_images(source, step), (cols, rows))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(
        f"Focal length {cal.focal_px:.1f} px at {cal.size[0]}x{cal.size[1]} "
        f"(horizontal field of view {cal.hfov_deg:.1f} deg); {cal.views} views, "
        f"reprojection error {cal.rms_px:.2f} px"
    )
    if save:
        console.print(f"Saved FINDOC_FOCAL_PX to {save_focal(cal.focal_px)}")
    else:
        console.print(f"To use it: set FINDOC_FOCAL_PX={cal.focal_px:.1f} (or rerun with --save)")


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


tamper_app = typer.Typer(help="Tamper detection model (Phase 6).", no_args_is_help=True)
app.add_typer(tamper_app, name="tamper")


@tamper_app.command("train")
def tamper_train(
    out: Path = typer.Option(Path("models/tamper.pkl"), "--out", "-o", help="Model file to write"),
    samples: int = typer.Option(80, help="Clean + forged training cheque pairs"),
    val_samples: int = typer.Option(40, "--val-samples", help="Validation pairs"),
    workers: int = typer.Option(-1, help="Processes rendering captures; -1 = CPU cores - 1"),
    seed: int = typer.Option(0),
) -> None:
    """Train the tamper classifier on synthetic clean and forged cheques (CPU, minutes)."""
    from findoc.fraud.tamper_train import train

    model = train(out, samples, val_samples, seed, workers, log=console.print)
    m = model.meta
    console.print(
        f"Saved {out}: forged cheques caught {m['detection_rate']:.0%}, clean cheques flagged "
        f"{m['false_alarm_rate']:.0%} (threshold {m['threshold']}), AUC {m['image_auc']}, "
        f"edited field named correctly {m['top_field_correct']:.0%}; {m['seconds']:.0f} s"
    )


@tamper_app.command("check")
def tamper_check(
    image: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(Path("outputs/tamper_check.jpg"), "--out", "-o"),
) -> None:
    """Tamper heatmap of a whole image (no page detection): what the detector sees."""
    from findoc.config import get_settings
    from findoc.fraud.tamper import classical_heatmap, load_model, upsample
    from findoc.fraud.viz import tamper_overlay
    from findoc.io import load_image

    img = load_image(image)
    page = np.ones(img.shape[:2], bool)
    model = load_model(get_settings().tamper_model)
    grid = model.block_probability(img, page) if model else classical_heatmap(img, page)
    heat = upsample(grid, img.shape[:2])
    save_image(out, tamper_overlay(img, heat))
    kind = "model" if model else "classical JPEG ghost (no model trained)"
    console.print(f"{kind}: hottest 1% of blocks {np.percentile(grid, 99):.2f}; saved {out}")


index_app = typer.Typer(help="Duplicate-cheque index (Phase 6).", no_args_is_help=True)
app.add_typer(index_app, name="index")


@index_app.command("list")
def index_list() -> None:
    """List the cheques in the duplicate index."""
    from findoc.config import get_settings
    from findoc.fraud.duplicates import DuplicateIndex

    idx = DuplicateIndex(get_settings().duplicate_index)
    t = Table(title=f"Duplicate index ({idx.root}): {len(idx)} cheques")
    for col in ("Entry", "Source", "Bank", "MICR"):
        t.add_column(col)
    for e in idx.entries:
        t.add_row(e.id, e.source, e.template_id, e.micr or "")
    console.print(t)


@index_app.command("clear")
def index_clear(
    yes: bool = typer.Option(False, "--yes", help="Don't ask for confirmation"),
) -> None:
    """Delete every entry of the duplicate index."""
    import shutil

    from findoc.config import get_settings

    root = get_settings().duplicate_index
    if not root.exists():
        console.print("The index is already empty.")
        return
    if not yes and not typer.confirm(f"Delete the duplicate index in {root}?"):
        raise typer.Exit(1)
    shutil.rmtree(root)
    console.print(f"Deleted {root}")


signatures_app = typer.Typer(
    help="Reference signatures per account (Phase 6).", no_args_is_help=True
)
app.add_typer(signatures_app, name="signatures")


@signatures_app.command("add")
def signatures_add(
    account: str = typer.Argument(..., help="Account number"),
    cheques: list[Path] = typer.Argument(..., exists=True, dir_okay=False, help="Cheque photos"),
) -> None:
    """Store the signature of each cheque photo as a reference for ACCOUNT."""
    from findoc.config import get_settings
    from findoc.fraud.signature import SignatureStore, signature_image
    from findoc.localize.templates import default_store
    from findoc.pipeline import default_pipeline

    s = get_settings()
    store = SignatureStore(s.signatures_dir)
    pipe = default_pipeline(ocr=False)
    pipe.stages = [
        st
        for st in pipe.stages
        if st[0] not in {"tamper", "duplicate", "signature", "rules", "risk"}
    ]
    for path in cheques:
        doc = pipe.run(path)
        if doc.registered is None or doc.template_id is None:
            console.print(f"[yellow]{path.name}: no cheque template matched; skipped[/]")
            continue
        sig = signature_image(doc.registered, default_store(s.templates_dir).get(doc.template_id))
        if sig is None:
            console.print(f"[yellow]{path.name}: no signature found; skipped[/]")
            continue
        console.print(f"{path.name}: saved {store.add(account, sig)}")


@signatures_app.command("list")
def signatures_list() -> None:
    """Accounts with reference signatures."""
    from findoc.config import get_settings
    from findoc.fraud.signature import SignatureStore

    store = SignatureStore(get_settings().signatures_dir)
    accounts = store.accounts()
    if not accounts:
        console.print(f"No reference signatures in {store.root}")
    for acc, n in accounts.items():
        console.print(f"{acc}: {n} reference(s)")


@app.command("export")
def export_cmd(
    results: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, help="Result JSON files from `findoc process --out`"
    ),
    out: Path = typer.Option(Path("outputs/export.xlsx"), "--out", "-o", help="Workbook to write"),
) -> None:
    """Export processed documents to an Excel workbook (documents, fields, line items, checks)."""
    from findoc.export.excel import load_results, write_workbook

    path = write_workbook(load_results(results), out)
    console.print(f"Wrote {path} ({len(results)} documents)")


classify_app = typer.Typer(help="Document-type classifier (Phase 7).", no_args_is_help=True)
app.add_typer(classify_app, name="classify")


@classify_app.command("train")
def classify_train(
    out: Path = typer.Option(Path("models/doctype.pkl"), "--out", "-o"),
    per_class: int = typer.Option(150, "--per-class", help="Training pages per class"),
    workers: int = typer.Option(-1, help="Processes rendering pages; -1 = CPU cores - 1"),
) -> None:
    """Train the cheque / invoice / receipt / other classifier on synthetic pages (CPU)."""
    from findoc.classify.doctype import train

    m = train(out, per_class=per_class, workers=workers, log=console.print)
    console.print(
        f"Saved {out}: validation accuracy {m.meta['accuracy']:.1%} "
        f"({m.meta['val_per_class']} pages per class); {m.meta['seconds']:.0f} s"
    )


invoice_app = typer.Typer(help="Invoice field extraction (Phase 7).", no_args_is_help=True)
app.add_typer(invoice_app, name="invoice")


@invoice_app.command("train")
def invoice_train(
    out: Path = typer.Option(Path("models/invoice_layoutlmv3"), "--out", "-o"),
    base: str = typer.Option(
        "microsoft/layoutlmv3-base",
        help="Pre-trained model to fine-tune ('tiny' = offline test model)",
    ),
    samples: int = typer.Option(1500, help="Synthetic invoices per epoch"),
    epochs: int = typer.Option(2, help="Validation F1 plateaued after epoch 1 on an RTX 3050"),
    batch_size: int = typer.Option(4, "--batch-size"),
    device: str = typer.Option("auto", help="'auto', 'cuda' or 'cpu'"),
    workers: int = typer.Option(-1, help="Processes rendering invoices; -1 = CPU cores - 2"),
) -> None:
    """Fine-tune LayoutLMv3 to tag invoice fields (needs the [layout] extra; GPU recommended)."""
    try:
        from findoc.recognize.layout import train
    except ImportError as exc:
        raise typer.BadParameter(f'{exc}. Install with: pip install -e ".[layout]"') from exc
    rec = train(
        out,
        base,
        samples=samples,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        workers=workers,
        log=console.print,
    )
    b = rec["best"]
    console.print(
        f"Saved {out}: field F1 {b['f1']}, line-item cells {b['item_cell_accuracy']} "
        f"(epoch {b['epoch']}, {rec['device']})"
    )


# Phase 8: web service (the app package ships in the same wheel; FastAPI is the [api] extra)
from app.cli import api_app  # noqa: E402

app.add_typer(api_app, name="api")
