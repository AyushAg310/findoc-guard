"""Synthetic Indian cheques in a CTS-2010-like layout, with ground truth.

Real cheque images are scarce and sensitive, so we render our own. All banks, names and
numbers are **fictional**.

Layout (template coordinates, 8 px per mm on a 203 x 92 mm leaf = 1624 x 736 px), following
the CTS-2010 arrangement used by Indian banks:

    +------------------------------------------------------------------------------+
    | BANK NAME / branch / IFSC                          DATE [D][D][M][M][Y][Y][Y][Y]
    | Pay ______________________________________________________ OR BEARER          |
    | Rupees ___________________________________________________   +-------------+  |
    | __________________________________________________           | Rs. figures |  |
    | A/c No. ...........                                   +--------------------+  |
    |                                                       |   signature area   |  |
    |  "Please sign above"                                  +--------------------+  |
    |------------------------------- MICR band ------------------------------------|
    |          123456  400002001  123456  31                                         |
    +------------------------------------------------------------------------------+

Each bank differs in colour, security pattern, logo and small layout offsets, so template
identification (Phase 2) has something real to tell apart.

Variable content can be written in a TrueType handwriting font if one is configured (see
:func:`_find_handwriting_font`); otherwise OpenCV's Hershey script font is used.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from findoc.recognize.parsers import amount_in_words, indian_grouping

TEMPLATE_SIZE = (1624, 736)  # width, height in px (8 px/mm)
INK = (110, 35, 20)  # dark blue ballpoint, BGR
PRINT = (35, 35, 35)
FONT = cv2.FONT_HERSHEY_SIMPLEX
SCRIPT = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

Box = tuple[int, int, int, int]  # x0, y0, x1, y1


@dataclass(frozen=True)
class FieldSpec:
    """Where a field is on the template and how to read it."""

    name: str
    box: Box
    kind: str  # "handwritten" | "printed" | "signature"
    allowlist: str | None = None
    cells: int = 1  # >1 for boxed fields: one character per equal-width cell (date boxes)


@dataclass(frozen=True)
class BankStyle:
    code: str  # 4-letter IFSC bank code
    name: str
    tint: tuple[int, int, int]  # background colour (BGR)
    accent: tuple[int, int, int]  # logo/pattern colour (BGR)
    pattern_freq: float  # security-pattern wave frequency
    dx: int = 0  # layout offsets so banks differ slightly
    dy: int = 0
    logo: str = "circle"  # circle | diamond | bars | ring | shield | peak | square | wave
    pattern: str = "waves"  # security background: waves | rosette | mesh | dots
    branch: str = "Main Branch, Jaipur"


# Fictional banks only: rendering look-alikes of real banks' cheques is how forgeries start.
# Real banks are added as local templates (``findoc templates``, or the admin's template
# editor in the web UI), which stay on the machine that uses them.
BANKS: dict[str, BankStyle] = {
    "ARVL": BankStyle(
        "ARVL", "ARAVALLI BANK", (250, 238, 222), (150, 90, 30), 0.020, 0, 0, "circle"
    ),
    "INDV": BankStyle(
        "INDV", "INDUS VALLEY BANK", (226, 246, 228), (40, 120, 40), 0.031, 12, -6, "diamond"
    ),
    "KVRY": BankStyle(
        "KVRY", "KAVERI CO-OPERATIVE BANK", (236, 228, 250), (90, 40, 140), 0.026, -10, 8, "bars"
    ),
    "DCCN": BankStyle(
        "DCCN", "DECCAN COMMERCIAL BANK", (214, 242, 250), (30, 110, 170), 0.017, 6, 10, "ring"
    ),
    # added in v0.9.1: 8 more styles, so template identification has a real choice to make
    "VNDS": BankStyle("VNDS", "VINDHYA SETU BANK", (232, 240, 246), (60, 70, 150), 0.023, -6,
                      4, "shield", "rosette", "Civil Lines, Bhopal"),
    "SHYL": BankStyle("SHYL", "SAHYADRI LEDGER BANK", (222, 236, 222), (50, 100, 60), 0.028, 8,
                      -4, "peak", "mesh", "FC Road, Pune"),
    "KNPL": BankStyle("KNPL", "KONKAN PEARL BANK", (246, 238, 230), (140, 100, 40), 0.019, -12,
                      -8, "wave", "dots", "Station Road, Ratnagiri"),
    "MGVL": BankStyle("MGVL", "MEGHA VALLEY BANK", (240, 230, 238), (120, 50, 110), 0.033, 4, 6,
                      "square", "rosette", "Police Bazar, Shillong"),
    "THRS": BankStyle("THRS", "THAR SANDS BANK", (214, 232, 246), (40, 90, 160), 0.021, 10, 2,
                      "peak", "dots", "Station Road, Bikaner"),
    "NLGC": BankStyle("NLGC", "NILGIRI CREST BANK", (230, 244, 238), (70, 120, 90), 0.025, -8,
                      10, "shield", "mesh", "Commercial Road, Ooty"),
    "GNGT": BankStyle("GNGT", "GANGOTRI TRUST BANK", (244, 240, 226), (150, 110, 50), 0.018, 2,
                      -10, "wave", "waves", "Har Ki Pauri, Haridwar"),
    "BRHR": BankStyle("BRHR", "BRAHMAPUTRA HERITAGE BANK", (226, 228, 244), (60, 50, 130), 0.03,
                      -4, -2, "square", "rosette", "GS Road, Guwahati"),
}  # fmt: skip

DIGITS = "0123456789"


def field_specs(style: BankStyle) -> list[FieldSpec]:
    """Field boxes for a bank (base CTS layout shifted by the bank's offsets)."""
    dx, dy = style.dx, style.dy

    def b(x0: int, y0: int, x1: int, y1: int) -> Box:
        return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)

    return [
        FieldSpec("date", b(1172, 36, 1556, 100), "handwritten", DIGITS, cells=8),
        FieldSpec("payee", b(110, 196, 1290, 262), "handwritten"),
        FieldSpec("amount_words_1", b(170, 276, 1290, 342), "handwritten"),
        FieldSpec("amount_words_2", b(40, 352, 1090, 418), "handwritten"),
        FieldSpec("amount_figures", b(1386, 352, 1586, 420), "handwritten", DIGITS + ",./-"),
        FieldSpec("account_number", b(166, 442, 720, 494), "printed", DIGITS),
        FieldSpec("ifsc", b(140, 150, 520, 188), "printed", None),
        FieldSpec("signature", b(1100, 470, 1590, 600), "signature"),
        FieldSpec("micr", b(300, 664, 1360, 724), "printed", DIGITS + " "),
    ]


# --------------------------------------------------------------------------- drawing helpers


def _security_pattern(img: np.ndarray, style: BankStyle) -> None:
    """Faint anti-copy background, as printed on real cheques: guilloche waves, a rosette of
    rings, a diagonal mesh, or a dot screen."""
    h, w = img.shape[:2]
    colour = tuple(int(c * 0.35 + 255 * 0.65) for c in style.accent)
    faint = tuple(int(c * 0.18 + t * 0.82) for c, t in zip(style.accent, style.tint, strict=True))
    if style.pattern == "rosette":
        cx, cy = w // 2, h // 2
        for i, r in enumerate(range(60, int(np.hypot(w, h) / 2), 34)):
            cv2.ellipse(img, (cx, cy), (r, int(r * 0.55)), (i * 5) % 180, 0, 360, faint, 1,
                        cv2.LINE_AA)  # fmt: skip
        return
    if style.pattern == "mesh":
        step = int(14 + style.pattern_freq * 300)
        for k in range(-h, w, step):
            cv2.line(img, (k, 0), (k + h, h), colour, 1, cv2.LINE_AA)
            cv2.line(img, (k + h, 0), (k, h), colour, 1, cv2.LINE_AA)
        return
    if style.pattern == "dots":
        step = int(10 + style.pattern_freq * 200)
        for y in range(4, h, step):
            for x in range(4 + (y // step % 2) * step // 2, w, step):
                cv2.circle(img, (x, y), 2, colour, -1, cv2.LINE_AA)
        return
    xs = np.arange(0, w, 3)
    for k in range(0, h + 60, 22):
        ys = k + 9 * np.sin(xs * style.pattern_freq + k * 0.05)
        pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, colour, 1, cv2.LINE_AA)


def _logo(img: np.ndarray, style: BankStyle, x: int, y: int) -> None:
    c = style.accent
    if style.logo == "circle":
        cv2.circle(img, (x + 35, y + 35), 32, c, -1)
        cv2.putText(img, style.code[0], (x + 20, y + 50), FONT, 1.3, (255, 255, 255), 3)
    elif style.logo == "diamond":
        pts = np.array([[x + 35, y], [x + 70, y + 35], [x + 35, y + 70], [x, y + 35]])
        cv2.fillPoly(img, [pts], c)
    elif style.logo == "bars":
        for i in range(4):
            cv2.rectangle(img, (x + i * 18, y + 60 - i * 15), (x + i * 18 + 12, y + 70), c, -1)
    elif style.logo == "shield":
        pts = np.array([[x + 5, y + 4], [x + 65, y + 4], [x + 65, y + 38], [x + 35, y + 70],
                        [x + 5, y + 38]])  # fmt: skip
        cv2.fillPoly(img, [pts], c)
        cv2.putText(img, style.code[0], (x + 20, y + 46), FONT, 1.1, (255, 255, 255), 3)
    elif style.logo == "peak":
        cv2.fillPoly(img, [np.array([[x, y + 70], [x + 28, y + 12], [x + 48, y + 44],
                                     [x + 58, y + 30], [x + 72, y + 70]])], c)  # fmt: skip
    elif style.logo == "square":
        cv2.rectangle(img, (x + 4, y + 4), (x + 66, y + 66), c, -1)
        cv2.rectangle(img, (x + 16, y + 16), (x + 54, y + 54), (255, 255, 255), 4)
    elif style.logo == "wave":
        xs = np.arange(0, 70)
        for j in range(3):
            ys = y + 20 + j * 18 + 7 * np.sin(xs / 9.0)
            pts = np.stack([x + xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, c, 6, cv2.LINE_AA)
    else:
        cv2.circle(img, (x + 35, y + 35), 32, c, 6)
        cv2.circle(img, (x + 35, y + 35), 12, c, -1)


def render_blank(style: BankStyle) -> np.ndarray:
    """The pre-printed cheque leaf with no variable content: the template reference image."""
    w, h = TEMPLATE_SIZE
    dx, dy = style.dx, style.dy
    img = np.empty((h, w, 3), np.uint8)
    img[:] = style.tint
    _security_pattern(img, style)
    img[650:, :] = (255, 255, 255)  # MICR clear band must be plain white

    _logo(img, style, 40 + dx, 30 + dy)
    cv2.putText(img, style.name, (130 + dx, 82 + dy), FONT, 1.25, style.accent, 3, cv2.LINE_AA)
    cv2.putText(img, style.branch, (130 + dx, 120 + dy), FONT, 0.6, PRINT, 1, cv2.LINE_AA)
    cv2.putText(img, "IFSC:", (40 + dx, 178 + dy), FONT, 0.75, PRINT, 2, cv2.LINE_AA)

    # Date boxes DDMMYYYY
    cv2.putText(img, "DATE", (1100 + dx, 80 + dy), FONT, 0.6, PRINT, 1, cv2.LINE_AA)
    for i in range(8):
        x0 = 1172 + dx + i * 48
        cv2.rectangle(img, (x0, 40 + dy), (x0 + 48, 96 + dy), PRINT, 1)
    cv2.putText(img, "D D M M Y Y Y Y", (1182 + dx, 116 + dy), FONT, 0.45, PRINT, 1)

    # Pay line
    cv2.putText(img, "Pay", (40 + dx, 252 + dy), FONT, 0.9, PRINT, 2, cv2.LINE_AA)
    cv2.line(img, (110 + dx, 258 + dy), (1290 + dx, 258 + dy), PRINT, 1)
    cv2.putText(img, "OR BEARER", (1310 + dx, 252 + dy), FONT, 0.7, PRINT, 2, cv2.LINE_AA)

    # Rupees lines
    cv2.putText(img, "Rupees", (40 + dx, 332 + dy), FONT, 0.9, PRINT, 2, cv2.LINE_AA)
    cv2.line(img, (170 + dx, 338 + dy), (1290 + dx, 338 + dy), PRINT, 1)
    cv2.line(img, (40 + dx, 414 + dy), (1090 + dx, 414 + dy), PRINT, 1)

    # Amount figures box
    cv2.rectangle(img, (1330 + dx, 348 + dy), (1590 + dx, 424 + dy), PRINT, 2)
    cv2.putText(img, "Rs.", (1338 + dx, 398 + dy), FONT, 0.8, PRINT, 2, cv2.LINE_AA)

    # Account number label, signature box, notices
    cv2.putText(img, "A/c No.", (40 + dx, 482 + dy), FONT, 0.75, PRINT, 2, cv2.LINE_AA)
    cv2.putText(img, "Please sign above", (1250 + dx, 628 + dy), FONT, 0.6, PRINT, 1, cv2.LINE_AA)
    cv2.putText(img, "Valid for 3 months only", (40 + dx, 600 + dy), FONT, 0.55, PRINT, 1)
    cv2.line(img, (0, 650), (w, 650), (200, 200, 200), 1)
    return img


# --------------------------------------------------------------------------- handwriting


@lru_cache(maxsize=1)
def _find_handwriting_font() -> str | None:
    """A TTF handwriting font, from FINDOC_HANDWRITING_FONT or data/fonts/*.ttf."""
    env = os.environ.get("FINDOC_HANDWRITING_FONT")
    if env and Path(env).is_file():
        return env
    fonts = sorted(Path("data/fonts").glob("*.ttf")) if Path("data/fonts").is_dir() else []
    return str(fonts[0]) if fonts else None


def _write(
    img: np.ndarray,
    text: str,
    box: Box,
    rng: np.random.Generator,
    colour: tuple[int, int, int] = INK,
    font_path: str | None = None,
    printed: bool = False,
) -> None:
    """Write ``text`` inside ``box``, sized to fit, with slight hand jitter."""
    if not text:
        return
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if font_path and not printed:
        from PIL import Image, ImageDraw, ImageFont

        size = int(bh * 0.8)
        font = ImageFont.truetype(font_path, size)
        while font.getlength(text) > bw * 0.97 and size > 10:
            size -= 2
            font = ImageFont.truetype(font_path, size)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text((x0 + 4, y0 + int(rng.integers(0, 4))), text, font=font, fill=colour[::-1])
        img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        return
    font = FONT if printed else SCRIPT
    thickness = 2
    scale = (bh * 0.55) / cv2.getTextSize("Hg", font, 1.0, thickness)[0][1]
    while cv2.getTextSize(text, font, scale, thickness)[0][0] > bw * 0.97 and scale > 0.3:
        scale *= 0.95
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    jitter = 0 if printed else int(rng.integers(-3, 4))
    org = (x0 + 6, y0 + (bh + th) // 2 + jitter)
    cv2.putText(img, text, org, font, scale, colour, thickness, cv2.LINE_AA)


def _signature(img: np.ndarray, box: Box, rng: np.random.Generator) -> None:
    """A random scribble: a smooth curve through a few control points plus an underline."""
    x0, y0, x1, y1 = box
    n = 7
    xs = np.linspace(x0 + 60, x1 - 80, n) + rng.normal(0, 10, n)
    ys = (y0 + y1) / 2 + rng.normal(0, 22, n)
    t = np.linspace(0, n - 1, 200)
    px = np.interp(t, np.arange(n), xs)
    py = np.interp(t, np.arange(n), ys) + 10 * np.sin(t * 3.1)
    pts = np.stack([px, py], 1).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, INK, 3, cv2.LINE_AA)
    cv2.line(img, (int(xs[0]), int(ys.max()) + 18), (int(xs[-1]), int(ys.max()) + 14), INK, 2)


# --------------------------------------------------------------------------- filled cheques

FIRST = ["Ravi", "Priya", "Arjun", "Neha", "Karan", "Anjali", "Vikram", "Sneha", "Rahul", "Meera",
         "Aditya", "Pooja", "Rohan", "Divya", "Sanjay", "Kavya"]  # fmt: skip
LAST = ["Sharma", "Verma", "Gupta", "Iyer", "Reddy", "Nair", "Singh", "Patel", "Mehta", "Joshi",
        "Rao", "Das", "Kulkarni", "Bose"]  # fmt: skip
COMPANIES = ["Sunrise Traders", "Pink City Supplies", "Shree Ganesh Stores", "Apex Logistics",
             "Lotus Pharma", "Desert Rose Exports"]  # fmt: skip


@dataclass
class ChequeTruth:
    """Ground truth for one generated cheque."""

    bank: str
    ifsc: str
    payee: str
    amount: str  # Decimal as string
    amount_words: str  # what is written (may disagree with amount if mismatch=True)
    date: str  # ISO date written on the cheque
    account_number: str
    micr: str
    signed: bool
    fraud: dict[str, bool] = field(default_factory=dict)
    fields: dict[str, str] = field(default_factory=dict)  # expected text per field box


def _split_words(text: str, first_line_chars: int = 46) -> tuple[str, str]:
    words, line1 = text.split(), ""
    while words and len(line1) + len(words[0]) + 1 <= first_line_chars:
        line1 = (line1 + " " + words.pop(0)).strip()
    return line1, " ".join(words)


def random_amount(rng: np.random.Generator) -> Decimal:
    """Mostly round business amounts, spread over 100 rupees to 50 lakh."""
    magnitude = rng.choice([2, 3, 4, 5, 6], p=[0.1, 0.25, 0.3, 0.25, 0.1])
    value = int(rng.integers(1, 10)) * 10 ** int(magnitude)
    value += int(rng.integers(0, 10)) * 10 ** max(int(magnitude) - 1, 0)
    if rng.random() < 0.3:
        value += int(rng.integers(1, 999))
    return Decimal(value)


def render_cheque(
    rng: np.random.Generator,
    bank: str | None = None,
    today: date | None = None,
    mismatch: bool = False,
    stale: bool = False,
    postdated: bool = False,
    unsigned: bool = False,
    signer: object | None = None,
    skilled_forgery: bool = False,
) -> tuple[np.ndarray, ChequeTruth]:
    """Render a filled cheque. Fraud flags create the conditions Phase 2 rules must catch.

    ``signer`` (a :class:`findoc.synth.signature.Signer`): sign as this person (Phase 6);
    ``skilled_forgery``: someone else copying that signature. Default: a random scribble."""
    today = today or date.today()
    code = bank or str(rng.choice(sorted(BANKS)))
    style = BANKS[code]
    specs = {f.name: f for f in field_specs(style)}
    img = render_blank(style)
    font_path = _find_handwriting_font()

    amount = random_amount(rng)
    written_amount = amount
    if mismatch:  # words say something different from the figures (altered cheque)
        written_amount = amount * int(rng.choice([2, 10])) if rng.random() < 0.5 else amount + 1000
    words = amount_in_words(written_amount)
    line1, line2 = _split_words(words)

    if stale:
        cheque_date = today - timedelta(days=int(rng.integers(100, 300)))
    elif postdated:
        cheque_date = today + timedelta(days=int(rng.integers(5, 60)))
    else:
        cheque_date = today - timedelta(days=int(rng.integers(0, 60)))

    payee = (
        f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if rng.random() < 0.7
        else str(rng.choice(COMPANIES))
    )
    ifsc = code + "0" + "".join(str(d) for d in rng.integers(0, 10, 6))
    account = "".join(str(d) for d in rng.integers(0, 10, int(rng.choice([11, 12, 14]))))
    micr = " ".join(
        [
            "".join(str(d) for d in rng.integers(0, 10, 6)),
            "302" + str(rng.integers(100, 999)) + str(rng.integers(100, 999)),
            account[-6:],
            "10",
        ]
    )
    date_text = cheque_date.strftime("%d%m%Y")
    figures = indian_grouping(amount) + "/-"

    # Date: one digit per box
    x0, y0, x1, y1 = specs["date"].box
    for i, ch in enumerate(date_text):
        bx = x0 + i * 48
        _write(img, ch, (bx + 8, y0 + 6, bx + 42, y1 - 6), rng, font_path=font_path)
    _write(img, payee, specs["payee"].box, rng, font_path=font_path)
    _write(img, line1, specs["amount_words_1"].box, rng, font_path=font_path)
    _write(img, line2, specs["amount_words_2"].box, rng, font_path=font_path)
    _write(img, figures, specs["amount_figures"].box, rng, font_path=font_path)
    _write(img, account, specs["account_number"].box, rng, colour=PRINT, printed=True)
    _write(img, ifsc, specs["ifsc"].box, rng, colour=PRINT, printed=True)
    _write(img, micr, specs["micr"].box, rng, colour=(10, 10, 10), printed=True)
    if not unsigned:
        if signer is not None:
            signer.draw(img, specs["signature"].box, rng, skilled=skilled_forgery)
        else:
            _signature(img, specs["signature"].box, rng)

    truth = ChequeTruth(
        bank=code,
        ifsc=ifsc,
        payee=payee,
        amount=str(amount.quantize(Decimal("0.01"))),
        amount_words=words,
        date=cheque_date.isoformat(),
        account_number=account,
        micr=micr,
        signed=not unsigned,
        fraud={"mismatch": mismatch, "stale": stale, "postdated": postdated, "unsigned": unsigned},
        fields={
            "date": date_text,
            "payee": payee,
            "amount_words_1": line1,
            "amount_words_2": line2,
            "amount_figures": figures,
            "account_number": account,
            "ifsc": ifsc,
            "micr": micr,
        },
    )
    return img, truth


def truth_dict(truth: ChequeTruth) -> dict:
    return asdict(truth)
