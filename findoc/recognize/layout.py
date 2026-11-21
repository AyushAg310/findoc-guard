"""LayoutLMv3 invoice field tagging: training and inference (Phase 7, FR-EXT-04, CO-2).

    pip install -e ".[layout]"
    findoc invoice train                       # GPU: ~15-25 min; downloads layoutlmv3-base once

**The model.** LayoutLMv3 (Huang et al., 2022) is a transformer pre-trained on millions of
document pages. Each input token carries three things: its **text** (sub-word embedding), its
**position on the page** (the word's box, scaled to 0-1000, embedded per coordinate) and the
page **image** (16x16 patches, embedded like a vision transformer). Self-attention mixes all
three, so "1,29,492.00" can be recognised as an *amount* because of where it sits (right
column, under "Amount") and what it looks like, not only what it says.

**Token classification.** Fine-tuning adds one linear layer that tags each token with a BIO
label: ``B-TOTAL`` = first word of the total, ``I-VENDOR`` = a later word of the vendor name,
``O`` = not a field. Only each word's first sub-word token is scored (the rest get label -100,
ignored by the loss), so predictions map back one per word.

**Speed.** On a CUDA GPU the forward and backward passes run in fp16 mixed precision
(``torch.autocast`` with a ``GradScaler`` that keeps small fp16 gradients from rounding to
zero): about twice as fast on tensor cores and half the memory. Batches are padded only to
their longest invoice (``collate``), not to 512 tokens, and invoices are rendered by
``workers`` background processes so the GPU never waits for the CPU.

**Training data** is generated on the fly from :mod:`findoc.synth.invoice`: a new invoice for
every step, with the page image degraded like a photo (blur, noise, lighting, JPEG) and the
words given OCR-like noise (box jitter and character errors). Validation uses a fixed set of
other invoices and reports field-level F1 of the assembled result
(:func:`findoc.recognize.invoice.assemble`), the same score the rule-based extractor gets.

``base="tiny"`` builds a small randomly initialised LayoutLMv3 with a tokenizer trained on
synthetic invoice text, entirely offline: it exists so tests and machines without internet can
run the whole training and inference path. Its scores are not meant to be compared with the
pre-trained model's.
"""

from __future__ import annotations

import inspect
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from findoc.recognize.invoice import Invoice, assemble, field_counts, split_phrases, summarise
from findoc.synth.invoice import LABELS, Word, render_invoice
from findoc.types import BBox, OcrWord

BIO = ["O"] + [f"{p}-{lab}" for lab in LABELS[1:] for p in ("B", "I")]
LABEL2ID = {lab: i for i, lab in enumerate(BIO)}
MAX_LEN = 512


def bio_tags(words: list[Word]) -> list[str]:
    """Field labels -> BIO tags: a field's first word is B-, its following words I-. A new
    field starts when the label or the item row changes, or the next word is not right after
    the previous one on the same line."""
    tags = []
    prev = None
    for w in words:
        if w.label == "O":
            tags.append("O")
            prev = None
            continue
        same = (
            prev is not None
            and prev.label == w.label
            and prev.row == w.row
            and abs(prev.box[1] - w.box[1]) < 6
            and 0 <= w.box[0] - prev.box[2] < 40
        )
        tags.append(("I-" if same else "B-") + w.label)
        prev = w
    return tags


def norm_box(b: BBox | tuple, w: int, h: int) -> list[int]:
    x0, y0, x1, y1 = (b.x0, b.y0, b.x1, b.y1) if isinstance(b, BBox) else b
    clip = lambda v: int(max(0, min(1000, v)))  # noqa: E731
    return [clip(1000 * x0 / w), clip(1000 * y0 / h), clip(1000 * x1 / w), clip(1000 * y1 / h)]


def degrade(page: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A clean rendered page made to look like a flattened phone capture."""
    img = page.astype(np.float32)
    h, w = img.shape[:2]
    gx = np.linspace(0, 1, w)[None, :, None]
    img *= (0.8 + 0.2 * rng.random()) + 0.15 * (gx - 0.5) * rng.choice([-1, 1])
    img += rng.normal(0, rng.uniform(1, 5), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.3, 1.2)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 95))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def noisy_words(words: list[Word], rng: np.random.Generator, cer: float) -> list[OcrWord]:
    """Word-level OCR noise that keeps one output per input word (so labels stay aligned)."""
    from findoc.synth.invoice import OCR_CONFUSIONS

    out = []
    for w in words:
        text = (
            "".join(OCR_CONFUSIONS.get(ch, ch) if rng.random() < cer else ch for ch in w.text)
            or w.text
        )
        j = rng.normal(0, 2.0, 4)
        x0, y0, x1, y1 = (w.box[0] + j[0], w.box[1] + j[1], w.box[2] + j[2], w.box[3] + j[3])
        out.append(
            OcrWord(
                text=text,
                confidence=1.0,
                bbox=BBox(x0=x0, y0=y0, x1=max(x1, x0 + 1), y1=max(y1, y0 + 1)),
            )
        )
    return out


def synthetic_sample(seed: int) -> tuple[np.ndarray, list[OcrWord], list[str], object]:
    """(degraded page, noisy words, BIO tags, truth) for one synthetic invoice."""
    rng = np.random.default_rng(seed)
    fraud = None if rng.random() < 0.6 else str(rng.choice(["total", "lineitem", "taxsplit"]))
    page, truth, words = render_invoice(rng, fraud=fraud)
    cer = float(rng.choice([0.0, 0.0, 0.01, 0.03]))
    return degrade(page, rng), noisy_words(words, rng, cer), bio_tags(words), truth


# ------------------------------------------------------------------------------- model I/O


def _tokenizer_from_files(vocab: Path, merges: Path):  # type: ignore[no-untyped-def]  # noqa: ANN202
    from transformers import LayoutLMv3TokenizerFast

    params = inspect.signature(LayoutLMv3TokenizerFast.__init__).parameters
    if "vocab_file" in params:  # transformers 4.x
        return LayoutLMv3TokenizerFast(vocab_file=str(vocab), merges_file=str(merges))
    return LayoutLMv3TokenizerFast(vocab=str(vocab), merges=str(merges))  # 5.x


def tiny_processor_and_model(seed: int = 0):  # type: ignore[no-untyped-def]  # noqa: ANN201
    """Offline: a byte-level BPE tokenizer trained on synthetic invoice text and a small
    randomly initialised LayoutLMv3 (2 layers, width 64)."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import (
        LayoutLMv3Config,
        LayoutLMv3ForTokenClassification,
        LayoutLMv3ImageProcessorPil,
        LayoutLMv3Processor,
    )

    texts = []
    for i in range(60):
        _, _, words = render_invoice(np.random.default_rng((seed, 99, i)))
        texts.append(" ".join(w.text for w in words))
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(
        texts,
        vocab_size=3000,
        min_frequency=1,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    d = Path(tempfile.mkdtemp(prefix="findoc_tok_"))
    bpe.save_model(str(d))
    tok = _tokenizer_from_files(d / "vocab.json", d / "merges.txt")
    proc = LayoutLMv3Processor(LayoutLMv3ImageProcessorPil(apply_ocr=False), tok)
    cfg = LayoutLMv3Config(
        vocab_size=len(tok),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=MAX_LEN + 2,
        input_size=224,
        coordinate_size=12,
        shape_size=8,  # 4 x 12 + 2 x 8 = 64 = hidden size
        num_labels=len(BIO),
        id2label=dict(enumerate(BIO)),
        label2id=LABEL2ID,
    )
    return proc, LayoutLMv3ForTokenClassification(cfg)


def pretrained_processor_and_model(base: str):  # type: ignore[no-untyped-def]  # noqa: ANN201
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

    proc = LayoutLMv3Processor.from_pretrained(base, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        base, num_labels=len(BIO), id2label=dict(enumerate(BIO)), label2id=LABEL2ID
    )
    return proc, model


def encode(proc, image: np.ndarray, words: list[OcrWord], tags: list[str] | None = None):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN201
    from PIL import Image

    h, w = image.shape[:2]
    kwargs = {}
    if tags is not None:
        kwargs["word_labels"] = [LABEL2ID[t] for t in tags]
    return proc(
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
        [wd.text for wd in words],
        boxes=[norm_box(wd.bbox, w, h) for wd in words],
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
        **kwargs,
    )


class SyntheticInvoices:
    """Training data: a new synthetic invoice for every (epoch, index), encoded for the model.
    A module-level class so DataLoader worker processes can receive it (Windows starts workers
    by pickling the dataset)."""

    def __init__(self, proc, samples: int, seed: int) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001
        self.proc, self.samples, self.seed, self.epoch = proc, samples, seed, 0

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, i: int) -> dict:
        img, words, tags, _ = synthetic_sample(hash((self.seed, self.epoch, i)) % (2**31))
        enc = encode(self.proc, img, words, tags)
        return {k: v[0] for k, v in enc.items()}


def collate(batch: list[dict], pad_id: int) -> dict:
    """Pad a batch only to its longest sequence (not always to 512 tokens): invoices are
    ~250-400 tokens, so fixed padding wasted a third or more of every step."""
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    fill = {"input_ids": pad_id, "attention_mask": 0, "labels": -100, "bbox": 0}
    out: dict = {}
    for key in batch[0]:
        if key == "pixel_values":
            out[key] = torch.stack([b[key] for b in batch])
            continue
        rows = []
        for b in batch:
            t = b[key]
            pad = n - t.shape[0]
            if pad:
                shape = (pad, *t.shape[1:])
                t = torch.cat([t, torch.full(shape, fill[key], dtype=t.dtype)])
            rows.append(t)
        out[key] = torch.stack(rows)
    return out


def predict_tags(proc, model, image: np.ndarray, words: list[OcrWord]) -> list[str]:  # type: ignore[no-untyped-def]  # noqa: ANN001
    """One BIO tag per word (words beyond the 512-token window get "O")."""
    import torch

    if not words:
        return []
    enc = encode(proc, image, words)
    device = next(model.parameters()).device
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"),
    ):
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0].float()
    pred = logits.argmax(-1).cpu().numpy()
    tags = ["O"] * len(words)
    seen = set()
    for tok_i, wid in enumerate(enc.word_ids(0)):
        if wid is not None and wid not in seen:
            seen.add(wid)
            tags[wid] = BIO[int(pred[tok_i])]
    return tags


def tags_to_invoice(words: list[OcrWord], tags: list[str], method: str) -> Invoice:
    inv = assemble(
        [(w, t.split("-", 1)[1] if t != "O" else "O") for w, t in zip(words, tags, strict=True)]
    )
    inv.method = method
    return inv


@dataclass
class LayoutExtractor:
    """A fine-tuned model saved by ``train`` (``findoc invoice train``)."""

    model_dir: Path

    def __post_init__(self) -> None:
        import torch
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

        self.proc = LayoutLMv3Processor.from_pretrained(str(self.model_dir))
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(str(self.model_dir))
        self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def extract(self, words: list[OcrWord], image: np.ndarray) -> Invoice:
        words = split_phrases(words)
        return tags_to_invoice(
            words, predict_tags(self.proc, self.model, image, words), "layoutlmv3"
        )


# --------------------------------------------------------------------------------- training


def train(
    out: Path = Path("models/invoice_layoutlmv3"),
    base: str = "microsoft/layoutlmv3-base",
    samples: int = 1500,
    val_samples: int = 60,
    epochs: int = 2,
    batch_size: int = 4,
    lr: float | None = None,
    device: str = "auto",
    seed: int = 0,
    workers: int = -1,
    log=print,  # noqa: ANN001
) -> dict:
    import os
    from functools import partial

    import torch
    from torch.utils.data import DataLoader

    from findoc.synth.invoice import truth_normalized

    dev = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    proc, model = (
        tiny_processor_and_model(seed) if base == "tiny" else pretrained_processor_and_model(base)
    )
    model.to(dev)
    lr = lr or (1e-3 if base == "tiny" else 3e-5)

    ds = SyntheticInvoices(proc, samples, seed)
    if workers < 0:  # render invoices in parallel so the GPU is not left waiting
        workers = max(0, min(8, (os.cpu_count() or 2) - 2))
    amp = dev.type == "cuda"  # fp16 mixed precision: ~2x faster on tensor cores, half the memory
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    pad_id = proc.tokenizer.pad_token_id
    val = [synthetic_sample(10_000_000 + seed * 1000 + i) for i in range(val_samples)]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((samples + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, steps), pct_start=0.1
    )
    history, best = [], -1.0
    t0 = time.perf_counter()
    for ep in range(epochs):
        ds.epoch = ep
        model.train()
        losses = []
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            collate_fn=partial(collate, pad_id=pad_id),
        )
        for batch in loader:
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                loss = model(**{k: v.to(dev, non_blocking=True) for k, v in batch.items()}).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        model.eval()
        counts = np.zeros(5)
        for img, words, _, truth in val:
            inv = tags_to_invoice(words, predict_tags(proc, model, img, words), "layoutlmv3")
            counts += field_counts(inv.normalized(), truth_normalized(truth))
        score = summarise(counts)
        history.append(
            {
                "epoch": ep + 1,
                "loss": round(float(np.mean(losses)), 4),
                **score,
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        log(
            f"epoch {ep + 1}/{epochs}: loss {history[-1]['loss']}, field F1 {score['f1']}, "
            f"item cells {score['item_cell_accuracy']} ({history[-1]['seconds']:.0f} s so far)"
        )
        if score["f1"] + score["item_cell_accuracy"] > best:
            best = score["f1"] + score["item_cell_accuracy"]
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out))
            proc.save_pretrained(str(out))
    record = {
        "base": base,
        "samples": samples,
        "epochs": epochs,
        "device": str(dev),
        "history": history,
        "best": max(history, key=lambda h: h["f1"] + h["item_cell_accuracy"]),
    }
    (out / "training.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
