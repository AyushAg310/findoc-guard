"""Synthetic Indian GST tax invoices with exact word boxes and labels (Phase 7).

A GST tax invoice states who sells (vendor, with GSTIN), who buys (buyer, often with GSTIN),
an invoice number and date, line items (description, HSN code, quantity, rate, amount), the
taxable value (subtotal), the tax, and the total. The tax depends on where the parties are:

* same state (the first two GSTIN digits match): **CGST + SGST**, each half the GST rate;
* different states: **IGST** at the full rate.

``render_invoice`` draws one invoice on an A4 page at 150 dpi (1240 x 1754 px) in one of three
layouts (``LAYOUTS``), with varied wording ("Invoice No." / "Bill No." / "Invoice #", "Sub
Total" / "Taxable Value", ...), so an extractor cannot memorise positions or labels. Every word
drawn is recorded with its box and its **label** (``LABELS``; ``O`` = not a field), the ground
truth for training and scoring extractors; line-item words also carry their row number.

``fraud`` plants the errors the Phase 7 checks must catch: ``total`` (grand total off),
``lineitem`` (one amount is not quantity x rate), ``taxsplit`` (IGST on an intra-state sale or
CGST + SGST on an inter-state one), ``gstin`` (vendor GSTIN with a wrong check character).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

import cv2
import numpy as np

from findoc.recognize import gstin as gst

PAGE = (1240, 1754)  # A4 at 150 dpi (width, height)
LABELS = (
    "O", "VENDOR", "VENDOR_GSTIN", "BUYER", "BUYER_GSTIN", "INV_NO", "DATE",
    "ITEM_DESC", "ITEM_HSN", "ITEM_QTY", "ITEM_RATE", "ITEM_AMOUNT",
    "SUBTOTAL", "CGST", "SGST", "IGST", "TOTAL",
)  # fmt: skip
LAYOUTS = ("classic", "banner", "compact")
FRAUDS = ("total", "lineitem", "taxsplit", "gstin")
GST_RATES = (5, 12, 18, 28)
INK = (30, 30, 30)

VENDORS = ["Shree Balaji Traders", "Aravalli Steel Works", "Pink City Handicrafts",
           "Desert Rose Textiles", "Lotus Pharma Distributors", "Sunrise Electricals",
           "Ganesh Agro Industries", "Apex Office Supplies", "Thar Logistics", "Kaveri Foods",
           "Malabar Spices Co", "Deccan Hardware Mart", "Indus Paper Products"]  # fmt: skip
BUYERS = ["Mehta Enterprises", "Rao Constructions", "Joshi Medical Store", "Patel Brothers",
          "Iyer and Sons", "Nair Retail", "Gupta General Store", "Verma Auto Parts",
          "Kulkarni Bakers", "Bose Stationers", "Reddy Infra Projects"]  # fmt: skip
ITEMS = [("Steel Rod 12mm", "7214"), ("Cotton Fabric", "5208"), ("LED Bulb 9W", "8539"),
         ("Copper Wire", "7408"), ("A4 Paper Ream", "4802"), ("Printer Toner", "8443"),
         ("Basmati Rice 25kg", "1006"), ("Turmeric Powder", "0910"), ("Cement Bag 50kg", "2523"),
         ("PVC Pipe 2in", "3917"), ("Office Chair", "9401"), ("Paracetamol 500mg", "3004"),
         ("Ceiling Fan", "8414"), ("Wall Paint 20L", "3209"), ("Laptop Bag", "4202"),
         ("Brass Lamp", "7419"), ("Handloom Saree", "5007"), ("Ball Bearing", "8482")]  # fmt: skip
STATE_CODES = ("08", "07", "27", "29", "24", "09", "33", "19", "06", "23")
CITIES = {
    "08": "Jaipur", "07": "New Delhi", "27": "Mumbai", "29": "Bengaluru", "24": "Ahmedabad",
    "09": "Lucknow", "33": "Chennai", "19": "Kolkata", "06": "Gurugram", "23": "Indore",
}  # fmt: skip


def money(x: Decimal) -> str:
    """Indian grouping with paise: 123456.5 -> '1,23,456.50'."""
    x = Decimal(x).quantize(Decimal("0.01"))
    rupees, paise = divmod(int(abs(x) * 100), 100)
    s = str(rupees)
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ("-" if x < 0 else "") + ",".join([*groups, tail]) + f".{paise:02d}"


def r2(x: Decimal) -> Decimal:
    return Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@dataclass
class Word:
    text: str
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 on the page
    label: str = "O"
    row: int | None = None  # line-item row, for item words


@dataclass
class Item:
    desc: str
    hsn: str
    qty: int
    rate: str  # Decimal as string
    amount: str


@dataclass
class InvoiceTruth:
    layout: str
    vendor: str
    vendor_gstin: str
    buyer: str
    buyer_gstin: str
    invoice_no: str
    date: str  # ISO
    items: list[Item]
    gst_rate: int
    intra_state: bool
    subtotal: str
    cgst: str | None
    sgst: str | None
    igst: str | None
    total: str
    fraud: str | None = None
    fields: dict[str, str] = field(default_factory=dict)  # text as printed, per field

    def as_dict(self) -> dict:
        return asdict(self)


class Canvas:
    """Draws text word by word and records each word's box and label."""

    def __init__(self, font: int, scale: float) -> None:
        self.img = np.full((PAGE[1], PAGE[0], 3), 255, np.uint8)
        self.font, self.scale = font, scale
        self.words: list[Word] = []

    def size(self, text: str, scale: float | None = None, thick: int = 1) -> tuple[int, int]:
        (w, h), _ = cv2.getTextSize(text, self.font, scale or self.scale, thick)
        return w, h

    def text(
        self,
        text: str,
        x: int,
        y: int,
        label: str = "O",
        row: int | None = None,
        scale: float | None = None,
        thick: int = 1,
        align: str = "left",
    ) -> int:
        """Draw ``text`` with its baseline at ``y``; returns the x after it. ``align='right'``
        makes ``x`` the right edge."""
        s = scale or self.scale
        space = self.size(" ", s, thick)[0]
        if align == "right":
            x -= self.size(text, s, thick)[0]
        for i, wtext in enumerate(text.split()):
            w, h = self.size(wtext, s, thick)
            cv2.putText(self.img, wtext, (x, y), self.font, s, INK, thick, cv2.LINE_AA)
            self.words.append(Word(wtext, (x, y - h, x + w, y + max(2, h // 4)), label, row))
            x += w + (space if i < len(text.split()) - 1 else 0)
        return x

    def labelled(self, key: str, value: str, x: int, y: int, label: str, gap: int = 12) -> int:
        """'Key: value' with only the value labelled."""
        x = self.text(key, x, y)
        return self.text(value, x + gap, y, label)

    def line(self, x0: int, y0: int, x1: int, y1: int, t: int = 1) -> None:
        cv2.line(self.img, (x0, y0), (x1, y1), (90, 90, 90), t)

    def box(self, x0: int, y0: int, x1: int, y1: int, t: int = 1) -> None:
        cv2.rectangle(self.img, (x0, y0), (x1, y1), (90, 90, 90), t)


def _pan(rng: np.random.Generator) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    p = [str(rng.choice(list(letters))) for _ in range(3)] + [str(rng.choice(list("CFPHT")))]
    p += [str(rng.choice(list(letters)))]
    return (
        "".join(p)
        + "".join(str(d) for d in rng.integers(0, 10, 4))
        + str(rng.choice(list(letters)))
    )


def make_truth(rng: np.random.Generator, layout: str, fraud: str | None) -> InvoiceTruth:
    vs = str(rng.choice(STATE_CODES))
    intra = rng.random() < 0.55
    bs = vs if intra else str(rng.choice([s for s in STATE_CODES if s != vs]))
    vendor_gstin = gst.make(vs, _pan(rng), str(rng.choice(list("123"))))
    if fraud == "gstin":  # one PAN character changed: format still fine, check digit wrong
        k = int(rng.integers(2, 12))
        pool = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if k in (2, 3, 4, 5, 6, 11) else "0123456789"
        new = str(rng.choice([ch for ch in pool if ch != vendor_gstin[k]]))
        vendor_gstin = vendor_gstin[:k] + new + vendor_gstin[k + 1 :]
        if gst.validate(vendor_gstin)[0]:  # the change happened to keep the checksum valid
            nxt = gst.CHARS[(gst.CHARS.index(vendor_gstin[14]) + 1) % 36]
            vendor_gstin = vendor_gstin[:14] + nxt
    buyer_gstin = gst.make(bs, _pan(rng), "1")
    n = int(rng.integers(2, 8))
    picks = rng.choice(len(ITEMS), size=n, replace=False)
    items = []
    for k in picks:
        desc, hsn = ITEMS[int(k)]
        qty = int(rng.integers(1, 60))
        rate = r2(Decimal(int(rng.integers(20, 25000))) + Decimal(int(rng.choice([0, 50]))) / 100)
        items.append(Item(desc, hsn, qty, str(rate), str(r2(rate * qty))))
    if fraud == "lineitem":
        j = int(rng.integers(0, n))
        wrong = r2(Decimal(items[j].amount) * Decimal(str(rng.choice([1.1, 0.9, 10]))))
        items[j].amount = str(wrong)
    subtotal = r2(sum(Decimal(i.amount) for i in items))
    rate = int(rng.choice(GST_RATES))
    charge_intra = intra if fraud != "taxsplit" else not intra
    if charge_intra:
        half = r2(subtotal * Decimal(rate) / 200)
        cgst = sgst = half
        igst = None
        tax = half * 2
    else:
        igst = r2(subtotal * Decimal(rate) / 100)
        cgst = sgst = None
        tax = igst
    total = r2(subtotal + tax)
    if fraud == "total":
        total = r2(
            total + Decimal(int(rng.choice([1, -1]))) * Decimal(int(rng.integers(100, 9000)))
        )
    year = int(rng.choice([2025, 2026]))
    d = date(year, 4, 1) + timedelta(days=int(rng.integers(0, 330)))
    fy_start = d.year if d.month >= 4 else d.year - 1  # Indian financial year: April-March
    fy = f"{fy_start}-{(fy_start + 1) % 100:02d}"
    prefix = str(rng.choice(["INV", "TI", "GST", "SB"]))
    inv_no = f"{prefix}/{fy}/{int(rng.integers(1, 9999)):04d}"
    return InvoiceTruth(
        layout=layout,
        vendor=str(rng.choice(VENDORS)),
        vendor_gstin=vendor_gstin,
        buyer=str(rng.choice(BUYERS)),
        buyer_gstin=buyer_gstin,
        invoice_no=inv_no,
        date=d.isoformat(),
        items=items,
        gst_rate=rate,
        intra_state=intra,
        subtotal=str(subtotal),
        cgst=None if cgst is None else str(cgst),
        sgst=None if sgst is None else str(sgst),
        igst=None if igst is None else str(igst),
        total=str(total),
        fraud=fraud,
    )


def _date_text(d: date, rng: np.random.Generator) -> str:
    return d.strftime(str(rng.choice(["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y"])))


def render_invoice(
    rng: np.random.Generator, layout: str | None = None, fraud: str | None = None
) -> tuple[np.ndarray, InvoiceTruth, list[Word]]:
    """One synthetic GST invoice: (page image, truth, every word with box and label)."""
    layout = layout or str(rng.choice(LAYOUTS))
    t = make_truth(rng, layout, fraud)
    font = int(rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX]))
    c = Canvas(font, float(rng.uniform(0.62, 0.72)))
    w, h = PAGE
    m = int(rng.integers(70, 100))  # margin
    d = date.fromisoformat(t.date)
    dtext = _date_text(d, rng)
    inv_key = str(rng.choice(["Invoice No.", "Invoice No:", "Bill No.", "Invoice #"]))
    date_key = str(rng.choice(["Date:", "Invoice Date:", "Dated:"]))
    t.fields = {"invoice_no": t.invoice_no, "date": dtext}
    vcity, bcity = CITIES[t.vendor_gstin[:2]], CITIES[t.buyer_gstin[:2]]
    sub_key = str(rng.choice(["Sub Total", "Taxable Value", "Total Taxable Amount"]))
    tot_key = str(rng.choice(["Grand Total", "Total Amount", "Invoice Total", "Amount Payable"]))

    # ------------------------------------------------------------------ header and parties
    if layout == "banner":
        c.box(m, 60, w - m, 150, 2)
        c.text("TAX INVOICE", w // 2 - c.size("TAX INVOICE", 1.3, 2)[0] // 2, 122, scale=1.3,
               thick=2)  # fmt: skip
        y = 220
        c.text(t.vendor, m, y, "VENDOR", scale=1.0, thick=2)
        c.text(f"Main Road, {vcity}", m, y + 40)
        c.labelled("GSTIN:", t.vendor_gstin, m, y + 80, "VENDOR_GSTIN")
        c.box(w - m - 470, 190, w - m, 330)
        c.labelled(inv_key, t.invoice_no, w - m - 450, 240, "INV_NO")
        c.labelled(date_key, dtext, w - m - 450, 290, "DATE")
        y = 400
        c.text("Bill To:", m, y, thick=2)
        c.text(t.buyer, m, y + 40, "BUYER")
        c.text(f"{bcity}", m, y + 80)
        c.labelled("GSTIN:", t.buyer_gstin, m, y + 120, "BUYER_GSTIN")
        top = 600
    elif layout == "compact":
        c.text(t.vendor, m, 110, "VENDOR", scale=1.15, thick=2)
        c.text(f"Industrial Area, {vcity}", m, 150)
        c.labelled("GSTIN", t.vendor_gstin, m, 190, "VENDOR_GSTIN")
        c.text("TAX INVOICE", w - m - c.size("TAX INVOICE", 0.9, 2)[0], 110, scale=0.9, thick=2)
        c.line(m, 220, w - m, 220, 2)
        c.labelled(inv_key, t.invoice_no, m, 270, "INV_NO")
        c.labelled(date_key, dtext, w // 2 + 40, 270, "DATE")
        c.labelled("Buyer:", t.buyer, m, 320, "BUYER")
        c.labelled("Buyer GSTIN:", t.buyer_gstin, w // 2 + 40, 320, "BUYER_GSTIN")
        c.line(m, 350, w - m, 350)
        top = 400
    else:  # classic
        c.text("TAX INVOICE", m, 100, scale=1.1, thick=2)
        c.text(t.vendor, m, 160, "VENDOR", scale=0.9, thick=2)
        c.text(f"Plot 14, {vcity}", m, 200)
        c.labelled("GSTIN:", t.vendor_gstin, m, 240, "VENDOR_GSTIN")
        x = w - m - 430
        c.labelled(inv_key, t.invoice_no, x, 160, "INV_NO")
        c.labelled(date_key, dtext, x, 200, "DATE")
        c.text(f"Place of Supply: {gst.STATES[t.buyer_gstin[:2]]}", x, 240)
        c.box(m, 290, w - m, 440)
        c.text("Details of Receiver", m + 15, 325, thick=2)
        c.text(t.buyer, m + 15, 365, "BUYER")
        c.labelled("GSTIN:", t.buyer_gstin, m + 15, 405, "BUYER_GSTIN")
        top = 500

    # --------------------------------------------------------------------------- the table
    cols = [("S.No", 0.0), ("Description", 0.07), ("HSN", 0.47), ("Qty", 0.58),
            ("Rate", 0.70), ("Amount", 0.86)]  # fmt: skip
    tw = w - 2 * m
    xs = [m + int(f * tw) for _, f in cols] + [w - m]
    rh = int(rng.integers(44, 54))
    grid = layout != "compact"
    c.line(m, top, w - m, top)
    for (name, _), x0, x1 in zip(cols, xs, xs[1:], strict=False):
        if name in ("Qty", "Rate", "Amount"):
            c.text(name, x1 - 10, top + rh - 16, thick=2, align="right")
        else:
            c.text(name, x0 + 8, top + rh - 16, thick=2)
    y = top + rh
    c.line(m, y, w - m, y)
    labels = ("O", "ITEM_DESC", "ITEM_HSN", "ITEM_QTY", "ITEM_RATE", "ITEM_AMOUNT")
    for r, it in enumerate(t.items):
        vals = (str(r + 1), it.desc, it.hsn, str(it.qty), money(Decimal(it.rate)),
                money(Decimal(it.amount)))  # fmt: skip
        for k, (val, lab) in enumerate(zip(vals, labels, strict=True)):
            if k >= 3:
                c.text(val, xs[k + 1] - 10, y + rh - 16, lab, row=r, align="right")
            else:
                c.text(val, xs[k] + 8, y + rh - 16, lab, row=r)
        y += rh
        if grid:
            c.line(m, y, w - m, y)
    if not grid:
        c.line(m, y, w - m, y)
    if grid:
        for x in xs:
            c.line(x, top, x, y)

    # ---------------------------------------------------------------------------- totals
    y += 60
    kx, vx = w - m - 520, w - m - 10
    if layout == "compact":
        kx = m
    rows = [(sub_key, t.subtotal, "SUBTOTAL")]
    half = Decimal(t.gst_rate) / 2
    if t.cgst is not None:
        rows += [(f"CGST @ {half:g}%", t.cgst, "CGST"), (f"SGST @ {half:g}%", t.sgst, "SGST")]
    else:
        rows += [(f"IGST @ {t.gst_rate}%", t.igst, "IGST")]
    rows += [(tot_key, t.total, "TOTAL")]
    for key, val, lab in rows:
        bold = 2 if lab == "TOTAL" else 1
        c.text(key, kx, y, thick=bold)
        txt = ("Rs. " if lab == "TOTAL" and rng.random() < 0.5 else "") + money(Decimal(val))
        c.text(txt, vx, y, lab if not txt.startswith("Rs.") else "O", thick=bold, align="right")
        if txt.startswith("Rs."):  # only the number is the field value
            c.words[-1].label = lab
        y += 44
    t.fields |= {"subtotal": money(Decimal(t.subtotal)), "total": money(Decimal(t.total))}
    y += 30
    c.text("Amount in words:", m, y)
    from findoc.recognize.parsers import amount_in_words

    c.text("Rupees " + amount_in_words(abs(Decimal(t.total))), m, y + 40, scale=c.scale * 0.9)
    c.text(f"For {t.vendor}", w - m - 420, h - 220)
    c.text("Authorised Signatory", w - m - 420, h - 120)
    c.line(m, h - 80, w - m, h - 80)
    c.text("This is a computer generated invoice.", m, h - 50, scale=0.5)
    return c.img, t, c.words


def labels_by_value(words: list[Word]) -> dict[str, str]:
    """Field label -> its words joined (line items excluded), as an extractor should return."""
    out: dict[str, list[str]] = {}
    for wd in words:
        if wd.label != "O" and wd.row is None:
            out.setdefault(wd.label, []).append(wd.text)
    return {k: " ".join(v) for k, v in out.items()}


def truth_normalized(t: InvoiceTruth) -> dict:
    """The truth in the extractors' normalised form (see findoc.recognize.invoice)."""
    from findoc.recognize.invoice import normalize_field

    f = {
        "vendor": t.vendor,
        "vendor_gstin": t.vendor_gstin,
        "buyer": t.buyer,
        "buyer_gstin": t.buyer_gstin,
        "invoice_no": t.invoice_no,
        "date": t.date,
        "subtotal": t.subtotal,
        "cgst": t.cgst,
        "sgst": t.sgst,
        "igst": t.igst,
        "total": t.total,
    }
    fields = {k: normalize_field(k, v) for k, v in f.items() if v is not None}
    fields["date"] = t.date
    items = [
        {"desc": normalize_field("desc", i.desc), "hsn": i.hsn, "qty": str(i.qty),
         "rate": i.rate, "amount": i.amount}
        for i in t.items
    ]  # fmt: skip
    return {"fields": fields, "items": items}


OCR_CONFUSIONS = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5", "8": "B", "B": "8",
                  ",": ".", ".": ",", "2": "Z", "6": "G", "rn": "m"}  # fmt: skip


def simulate_ocr(
    words: list[Word],
    rng: np.random.Generator,
    cer: float = 0.0,
    merge: float = 0.5,
    jitter: float = 2.0,
) -> list:
    """Ground-truth words -> OCR-like output: neighbouring words on a line merged into phrases
    (EasyOCR reads phrases) with probability ``merge``, boxes jittered by ``jitter`` px, and a
    character error rate ``cer`` of typical confusions. Returns (OcrWord list, labels per
    original word) for evaluation; labels refer to the unmerged words."""
    from findoc.types import BBox, OcrWord

    out = []
    i = 0
    ws = sorted(words, key=lambda w: (round((w.box[1] + w.box[3]) / 2 / 8), w.box[0]))
    while i < len(ws):
        group = [ws[i]]
        while (i + len(group) < len(ws) and rng.random() < merge
               and abs(ws[i + len(group)].box[1] - group[-1].box[1]) < 6
               and 0 <= ws[i + len(group)].box[0] - group[-1].box[2] < 40):  # fmt: skip
            group.append(ws[i + len(group)])
        text = " ".join(g.text for g in group)
        if cer > 0:
            chars = []
            for ch in text:
                if ch != " " and rng.random() < cer:
                    chars.append(OCR_CONFUSIONS.get(ch, ch if rng.random() < 0.5 else ""))
                else:
                    chars.append(ch)
            text = "".join(chars) or group[0].text
        x0 = min(g.box[0] for g in group) + rng.normal(0, jitter)
        y0 = min(g.box[1] for g in group) + rng.normal(0, jitter)
        x1 = max(g.box[2] for g in group) + rng.normal(0, jitter)
        y1 = max(g.box[3] for g in group) + rng.normal(0, jitter)
        box = BBox(x0=x0, y0=y0, x1=max(x1, x0 + 1), y1=max(y1, y0 + 1))
        out.append(OcrWord(text=text, confidence=1.0 - cer, bbox=box))
        i += len(group)
    return out


def invoice_photo(
    seed: int, fraud: str | None = None, blur_sigma: float = 0.0
) -> tuple[np.ndarray, InvoiceTruth, list[Word], np.ndarray]:
    """A phone photo of a synthetic invoice: (photo, truth, words on the page, page -> photo
    homography)."""
    from findoc.synth.photo import photograph

    rng = np.random.default_rng(seed)
    page, truth, words = render_invoice(rng, fraud=fraud)
    photo, quad = photograph(page, rng, size=(1500, 2000), fill=(0.5, 0.75), blur_sigma=blur_sigma)
    h, w = page.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    return photo, truth, words, cv2.getPerspectiveTransform(src, quad.astype(np.float32))


def generate_invoices(
    n: int, out_dir, seed: int = 0, fraud_rate: float = 0.4, blur_sigma: float = 0.0
) -> list:  # noqa: ANN001
    """Write ``n`` invoice photos and labels.json (truth, words with labels, homography)."""
    import json
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    picker = np.random.default_rng(seed + 31_337)
    labels, paths = {}, []
    for i in range(n):
        fraud = str(picker.choice(FRAUDS)) if picker.random() < fraud_rate else None
        photo, truth, words, hmat = invoice_photo(seed + i, fraud, blur_sigma)
        name = f"invoice_{seed + i:04d}.jpg"
        cv2.imwrite(str(out_dir / name), photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
        labels[name] = truth.as_dict() | {
            "page_to_photo": hmat.round(6).tolist(),
            "words": [[w.text, list(w.box), w.label, w.row] for w in words],
        }
        paths.append(out_dir / name)
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    return paths
