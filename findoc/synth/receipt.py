"""Synthetic till receipts (thermal-printer style), for document-type classification (Phase 7).

A receipt is narrow (58 or 80 mm paper: ~2.5-4.5x taller than wide), monospaced, centred
header, dashed separators, one line per item with the price right-aligned, and a total.
"""

from __future__ import annotations

import cv2
import numpy as np

STORES = [
    "Sharma Kirana Store",
    "City Mart",
    "Annapurna Sweets",
    "Green Leaf Cafe",
    "Metro Pharmacy",
    "Royal Bakers",
    "Fresh Basket",
    "Chai Point Express",
]
ITEMS = [
    "Milk 1L",
    "Bread",
    "Eggs 12",
    "Rice 5kg",
    "Atta 10kg",
    "Sugar 1kg",
    "Tea 250g",
    "Paneer 200g",
    "Biscuits",
    "Soap",
    "Shampoo",
    "Coffee",
    "Samosa",
    "Masala Dosa",
    "Paracetamol",
    "Notebook",
]


def render_receipt(rng: np.random.Generator) -> np.ndarray:
    w = int(rng.choice([384, 576]))  # 58 mm / 80 mm paper at 203 dpi
    n = int(rng.integers(3, 14))
    h = max(330 + 34 * n + int(rng.integers(120, 260)), int(w * rng.uniform(2.4, 3.2)))
    img = np.full((h, w, 3), int(rng.integers(235, 252)), np.uint8)
    font = cv2.FONT_HERSHEY_PLAIN
    ink = (40, 40, 40)
    scale = 1.25 if w == 576 else 1.0

    def centred(text: str, y: int, s: float = scale, t: int = 1) -> None:
        (tw, _), _ = cv2.getTextSize(text, font, s, t)
        cv2.putText(img, text, ((w - tw) // 2, y), font, s, ink, t, cv2.LINE_AA)

    def dashes(y: int) -> None:
        for x in range(10, w - 10, 14):
            cv2.line(img, (x, y), (x + 7, y), ink, 1)

    centred(str(rng.choice(STORES)), 50, scale * 1.4, 2)
    centred("Jaipur", 85)
    centred(f"GSTIN 08{''.join(rng.choice(list('ABCDEFGHJK'), 5))}1234X1Z{rng.integers(0, 9)}", 115)
    centred(
        f"Bill {rng.integers(1000, 9999)}  {rng.integers(1, 28):02d}/{rng.integers(1, 12):02d}/26",
        145,
    )
    dashes(170)
    y = 205
    total = 0
    for _ in range(n):
        item = str(rng.choice(ITEMS))
        qty = int(rng.integers(1, 5))
        price = int(rng.integers(10, 600))
        total += qty * price
        cv2.putText(img, f"{item} x{qty}", (12, y), font, scale, ink, 1, cv2.LINE_AA)
        txt = f"{qty * price:.2f}"
        (tw, _), _ = cv2.getTextSize(txt, font, scale, 1)
        cv2.putText(img, txt, (w - 12 - tw, y), font, scale, ink, 1, cv2.LINE_AA)
        y += 34
    dashes(y - 10)
    y += 30
    cv2.putText(img, "TOTAL", (12, y), font, scale * 1.3, ink, 2, cv2.LINE_AA)
    txt = f"{total:.2f}"
    (tw, _), _ = cv2.getTextSize(txt, font, scale * 1.3, 2)
    cv2.putText(img, txt, (w - 12 - tw, y), font, scale * 1.3, ink, 2, cv2.LINE_AA)
    centred("THANK YOU! VISIT AGAIN", y + 70)
    return img
