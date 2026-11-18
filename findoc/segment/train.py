"""Train DocSegNet on synthetic samples and export it to ONNX (needs PyTorch).

    findoc segment train --samples 20000 --epochs 12          # GPU: ~15-25 min on an RTX 3050

Training data is generated on the fly (:mod:`findoc.synth.segdata`): every epoch sees new
backgrounds, poses and lighting, so the model cannot memorise images. Validation uses a fixed
set (different seeds) and reports mask **IoU** (intersection over union of predicted and true
document pixels).

Loss = binary cross-entropy + Dice loss. BCE judges every pixel independently; Dice
(1 - 2|P*T| / (|P| + |T|)) judges the overlap of the whole shape, which keeps a small document
from being ignored in favour of a large background.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from findoc.segment.model import INPUT_SIZE, DocSegNet, count_parameters
from findoc.synth.segdata import document_pool, make_sample


def to_input(img: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """BGR uint8 (any size) -> float32 (3, size, size) in [-0.5, 0.5]. Same as inference."""
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 - 0.5
    return x.transpose(2, 0, 1)


class SyntheticDocs(Dataset):
    def __init__(self, n: int, seed: int, pool_seed: int = 0, pool_size: int = 60) -> None:
        self.n, self.seed, self.epoch = n, seed, 0
        self.pool_seed, self.pool_size = pool_seed, pool_size
        self.pool: list[np.ndarray] | None = None

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pool is None:  # built lazily, once per worker process
            self.pool = document_pool(self.pool_seed, self.pool_size)
        rng = np.random.default_rng((self.seed, self.epoch, i))
        s = make_sample(rng, self.pool)
        m = cv2.resize(s.mask, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(to_input(s.image)), torch.from_numpy(
            (m > 127).astype(np.float32)[None]
        )


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()


def iou(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = (logits > 0).float()
    inter = (p * target).sum((1, 2, 3))
    union = ((p + target) > 0).float().sum((1, 2, 3))
    return inter / union.clamp(min=1)


@dataclass
class TrainResult:
    onnx_path: Path
    best_val_iou: float
    history: list[dict]
    parameters: int
    device: str
    seconds: float


def export_onnx(model: nn.Module, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    kw = {"dynamo": False} if "dynamo" in torch.onnx.export.__code__.co_varnames else {}
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        **kw,
    )
    return path


def train(
    out: Path = Path("models/docseg.onnx"),
    samples: int = 20000,
    val_samples: int = 400,
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 2e-3,
    workers: int = 2,
    device: str = "auto",
    seed: int = 0,
    log=print,  # noqa: ANN001
) -> TrainResult:
    torch.manual_seed(seed)
    dev = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else ("cpu" if device == "auto" else device)
    )
    model = DocSegNet().to(dev)
    train_set = SyntheticDocs(samples, seed=seed + 1)
    val_set = SyntheticDocs(val_samples, seed=10_000 + seed, pool_seed=999)  # unseen documents
    loader = DataLoader(
        train_set,
        batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=True,
    )
    val_loader = DataLoader(val_set, batch_size, num_workers=workers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader))
    bce = nn.BCEWithLogitsLoss()
    best, history, t0 = -1.0, [], time.time()
    for epoch in range(epochs):
        # Fresh samples every epoch. Workers are started anew for each epoch (no
        # persistent_workers), so they see the new epoch number; persistent workers would keep
        # their own copy of the dataset and replay epoch 0's images forever.
        train_set.epoch = epoch
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            ious = torch.cat([iou(model(x.to(dev)), y.to(dev)).cpu() for x, y in val_loader])
        val_iou = float(ious.mean())
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "val_iou": val_iou,
                "val_iou_p05": float(ious.quantile(0.05)),
            }
        )
        h = history[-1]
        log(
            f"epoch {epoch + 1}/{epochs}  loss {h['loss']:.4f}  val IoU {val_iou:.4f} "
            f"(5th pct {h['val_iou_p05']:.4f})  {time.time() - t0:.0f}s"
        )
        if val_iou > best:
            best = val_iou
            export_onnx(model, out)
            model.to(dev)
    meta = {
        "input_size": INPUT_SIZE,
        "normalise": "x/255 - 0.5, BGR",
        "best_val_iou": best,
        "history": history,
        "parameters": count_parameters(model),
        "samples": samples,
        "epochs": epochs,
        "device": str(dev),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return TrainResult(out, best, history, count_parameters(model), str(dev), time.time() - t0)
