# Phase 7 - Invoices: complete code explanation

Version **0.8.0** · Course outcome **CO-2** (a transformer fine-tuned for token classification, a
learned page classifier) with **CO-1/CO-3** support (page geometry, table structure from word
boxes), plus domain rules: GSTIN check digits and GST arithmetic.

## Why this phase exists

Cheques have one layout per bank, so Phase 2 could find every field by aligning a template.
Invoices have no fixed layout: every vendor's billing software prints its own. Fields have to be
found from **what the words say and where they are**, and then checked against the rules of
Indian GST:

| Problem on a GST invoice | Check |
| --- | --- |
| Total or a line amount doesn't add up | `INVOICE_MATH` |
| CGST + SGST charged between states, or IGST within one | `TAX_TYPE_MISMATCH` |
| A GSTIN that cannot exist (wrong check character) | `GSTIN_INVALID` |
| Totals not readable | `INVOICE_UNVERIFIED` (a prompt to check by hand) |

```
photo -> (Phases 1-5: flat page) -> classify: cheque / invoice / receipt / other
      -> OCR words + boxes -> extract: rules or LayoutLMv3 -> fields + line items
      -> GSTIN, arithmetic (misread-aware), tax type -> risk score (Phase 6) -> Excel export
```

---

## 1. Big picture

| Step | Technique | File |
| --- | --- | --- |
| Synthetic GST invoices | 3 layouts, varied wording, valid GSTINs, intra/inter-state tax, exact word boxes and labels, planted errors | `findoc/synth/invoice.py` |
| Synthetic receipts | Narrow thermal-printer receipts | `findoc/synth/receipt.py` |
| GSTIN | Format, state code, mod-36 check character, OCR-confusion repair | `findoc/recognize/gstin.py` |
| Rule-based extraction | Key-value anchors, GSTIN pattern, largest text, table from header columns | `findoc/recognize/invoice.py` |
| LayoutLMv3 | Text + box + image transformer, BIO token classification, fine-tuned on synthetic invoices | `findoc/recognize/layout.py`, `findoc invoice train` |
| Checks | Arithmetic with misread detection, tax type, GSTIN | `findoc/fraud/invoice_rules.py` |
| Document type | 641 page features (shape, layout, colour, ruled lines), gradient-boosted trees | `findoc/classify/doctype.py`, `findoc classify train` |
| Pipeline | `classify` after `enhance`; `invoice` after `ocr` | `findoc/classify/stages.py`, `findoc/recognize/invoice_stage.py` |
| Excel | Documents, fields, line items, checks | `findoc/export/excel.py`, `findoc export` |
| Measure | Extraction F1 vs. OCR error rate, checks, classifier | `experiments/phase7_invoice_eval.py` |

New dependencies: openpyxl (core); an optional `[layout]` extra (PyTorch, transformers) for
LayoutLMv3.

---

## 2. How a run flows

`findoc process invoice.jpg` (with OCR installed):

1. Phases 1-5 produce the flat page (`doc.enhanced`).
2. **classify** (if `models/doctype.pkl` exists): invoice / receipt / cheque / other. An invoice
   or receipt skips cheque template matching.
3. **ocr** reads every word with its box.
4. **invoice**: runs if the page was classified as an invoice, or (without a classifier) if the
   text has at least two invoice cues ("Tax Invoice", GSTIN, HSN, CGST, IGST). Extraction uses
   LayoutLMv3 when `models/invoice_layoutlmv3/` exists, otherwise the rules
   (`FINDOC_INVOICE_EXTRACTOR=auto|rules|layoutlmv3`). A GSTIN that fails its check
   character gets a **second look**: its crop is enlarged and read again with only 0-9/A-Z
   allowed, and a valid reading close to the first replaces it (a message says so). Fields go
   to `doc.fields`, rows to `doc.line_items`, and the checks to `doc.checks`.
5. **risk** (Phase 6) scores all checks; the console shows the fields, a line-items table and
   the checks; `--out result.json` saves everything.

`findoc export results/*.json --out invoices.xlsx` collects any number of results into one
workbook.

---

## 3. File-by-file walkthrough

### 3.1 `findoc/recognize/gstin.py`

A GSTIN is `SS PPPPPPPPPP E Z C`: state code, the business's PAN (5 letters, 4 digits, 1 letter),
entity number, the letter Z, and a **check character**. The check (GSTN's algorithm, a Luhn
variant in base 36): map the first 14 characters to 0-35 (`0-9A-Z`), multiply them alternately by
1 and 2, add each product's base-36 digits (`p // 36 + p % 36`), and the check value is
`(36 - sum % 36) % 36`. Any single wrong character changes it.

- `validate`: format (regex), known state code, check character; returns the reason.
- `clean`: removes separators, then fixes OCR letter/digit confusions **by position** (the
  format says positions 1-2 are digits, 3-7 letters, ...). Two more since v0.8.2, both from
  the real-OCR run: the 14th character is always Z (EasyOCR read "1Z" as "12"), and a
  digit-like letter in the entity number is the digit (I -> 1; letters there only start
  after a business's ninth registration in a state).
- `misread_fixes`: valid GSTINs one known OCR confusion away; the invoice check uses it to say
  "probably misread" (confidence 0.3, which keeps the invoice out of review on its own)
  instead of "invalid". `CONFUSABLE` lists only what EasyOCR was seen to do (it printed O for
  Q, D or 0; 0 for 9; 2 for 7; 5 for 3). Every pair you add makes more real forgeries look
  like misreads: the first, symmetric table excused 27% of planted invalid GSTINs, this one
  8%.
- `make`, `state_of`: build valid numbers (synthetic data), name the state.

Verified on published example numbers (`27AAPFU0939F1ZV`, `29AAGCB7383J1Z4`).

### 3.2 `findoc/synth/invoice.py` (and `receipt.py`)

`make_truth` draws the content: vendor and buyer with valid GSTINs in a random state (55% the same
state), 2-7 line items (description, HSN code, quantity, rate, amount = quantity x rate), a GST
rate (5/12/18/28%), CGST + SGST (half each) within a state or IGST between states, total, invoice
number in the Indian financial-year style (`GST/2026-27/0457`). Planted errors (`FRAUDS`): `total`
off by 100-9000, `lineitem` with one amount x 1.1 / 0.9 / 10, `taxsplit` the wrong tax type,
`gstin` one character changed (check digit fails).

`render_invoice` draws it in one of three layouts (`classic`, `banner`, `compact`) with random
fonts, sizes, margins and wording ("Invoice No." / "Bill No." / "Invoice #", "Sub Total" /
"Taxable Value", "Grand Total" / "Amount Payable"...). The `Canvas` draws text **word by word**
and records every word's box and label (`LABELS`: VENDOR, VENDOR_GSTIN, BUYER, BUYER_GSTIN,
INV_NO, DATE, ITEM_DESC/HSN/QTY/RATE/AMOUNT with a row number, SUBTOTAL, CGST, SGST, IGST,
TOTAL, or O). `simulate_ocr` turns the true words into OCR-like output: neighbouring words merged
into phrases (EasyOCR reads phrases), boxes jittered, and a chosen character error rate of
typical confusions (0/O, 1/l, 5/S, 8/B, ","/"."). `invoice_photo` / `generate_invoices` make
photos (`findoc synth --kind invoice`). `render_receipt` draws narrow till receipts for the
classifier.

### 3.3 `findoc/recognize/invoice.py`: rules and shared pieces

- `split_phrases`: OCR phrases -> words, each with a share of the box proportional to its
  characters. `lines_of`: words -> text lines (vertical centres within half a word height).
  `segments`: a line cut where the gap exceeds 2.5 x text height (two columns on one line).
- `RuleExtractor.extract`:
  - **key: value** for invoice number and date: the words after the key on the same line, up to
    a wide gap. Keys are matched after undoing digit-for-letter confusions (`_keytext`).
  - **amounts**: on a "Sub Total" / "CGST" / "SGST" / "IGST" / "Grand Total" line, the right-most
    number (`numeric` accepts misread digits).
  - **GSTINs**: any 15-character run (across up to three OCR tokens) that matches the pattern
    after `clean`; one inside the buyer block ("Bill To", "Receiver", "Buyer"...) is the
    buyer's.
  - **vendor**: the tallest text segment in the top quarter that is not a title or a label.
  - **table**: the first line naming three of Description / HSN / Qty / Rate / Amount is the
    header. Numbers go to the numeric column whose header's **right** edge is nearest (numbers
    are right-aligned); text goes to the text column whose left edge it follows. A row counts if
    its amount is a number; rows stop at the subtotal line.
- `assemble`: labelled words -> fields and rows (used by LayoutLMv3, and by a test that feeds the
  true labels and gets back exactly the truth).
- `normalize_field`: canonical values: `parse_money` (paise always printed, so a final `,dd` is
  a decimal point misread as a comma; Indian grouping never ends in two digits), dates in four
  formats, HSN 4-8 digits, invoice number without spaces, GSTIN format.
- `amount_in_words_text`: the "Rupees ... Only" line, parsed with the Phase 2 words-to-number
  parser; both extractors add it, because the checks use it as a tie-breaker.
- `field_counts`, `summarise`: scoring (header precision, recall, F1; line-item cell accuracy).

### 3.4 `findoc/recognize/layout.py`: LayoutLMv3

LayoutLMv3 (Huang et al., 2022) is a transformer pre-trained on millions of document pages. Each
token has a text embedding, a **2D position embedding** (its word's box scaled to 0-1000; x, y,
width and height each embedded) and the page image enters as 16x16 patches like a vision
transformer. Self-attention mixes them, so an amount is recognised by what it says, where it is
(right column, under "Amount") and how it looks.

Fine-tuning adds one linear layer that tags every token with a **BIO** label (`B-TOTAL` first
word of the total, `I-VENDOR` a later word of the vendor, `O` none; `bio_tags`). A word split into
several sub-word tokens is scored on its first token only (the others get label -100, which the
loss ignores), so each word gets exactly one prediction (`predict_tags` uses `word_ids`).

`train` generates a new invoice every step (`synthetic_sample`: page degraded like a photo, words
with OCR noise), uses AdamW with a one-cycle learning rate and gradient clipping, validates on 60
fixed invoices by assembling fields and scoring them like the rules, and saves the best epoch
with `save_pretrained`. `LayoutExtractor` loads it for the pipeline.

Since v0.8.1 training is faster in three ways:
- **fp16 mixed precision** on CUDA (`torch.autocast`, with a `GradScaler` that scales the loss up
  so small fp16 gradients don't round to zero): tensor cores roughly double the speed and the
  activations take half the memory;
- **dynamic padding** (`collate`): each batch is padded to its longest invoice (~250-400
  tokens), not to the model's 512;
- **parallel rendering** (`SyntheticInvoices` in `--workers` DataLoader processes, defined at
  module level so Windows can pickle it into the workers): the GPU doesn't wait for the CPU.

`base="tiny"` builds a 2-layer, width-64 LayoutLMv3 with a byte-level BPE tokenizer trained on
synthetic invoice text, entirely offline. The development environment cannot download the
pre-trained weights, so this is how the training and inference code is tested; in a short CPU
run its field F1 rose from 0 to 0.40 in three epochs (the labels line up). Pre-trained results
come from your GPU run (section 5).

### 3.5 `findoc/fraud/invoice_rules.py`

- `check_gstins`: invalid -> `GSTIN_INVALID`, severity 0.7, confidence 0.9; but if one common OCR
  confusion makes it valid, confidence 0.3 and the reason says "probably misread".
- `check_math` uses the invoice's **redundancy**: most numbers can be derived two ways (an amount
  is also quantity x rate; the subtotal is also the sum of the lines; the total is also subtotal
  + tax and is written in words). A number that disagrees while the other derivation fits
  everything else is a **probable misread**: corrected and listed in the evidence, not reported:
  - a line where one commonly confused digit (in quantity, rate or amount) makes it add up;
  - a line amount whose quantity x rate makes the lines add up to the subtotal;
  - a subtotal when the lines plus tax equal the total;
  - CGST / SGST that differ when one of them fits a GST rate *and* the total;
  - a total when subtotal + tax and the amount in words agree.

  The line-sum test runs only when every line was read and is self-consistent (a damaged row
  would otherwise look like a wrong subtotal). Severity grows with the relative error (0.5 for a
  tiny one, 1.0 from 5%).
- `check_tax_type`: same state code -> CGST + SGST expected; different -> IGST.

Risk points (Phase 6): `INVOICE_MATH` 80, `GSTIN_INVALID` 60, `TAX_TYPE_MISMATCH` 60,
`INVOICE_UNVERIFIED` 30.

### 3.6 `findoc/classify/doctype.py` and `stages.py`

`page_features` (641 numbers): aspect ratio and its log; saturation mean/spread and share of
coloured pixels; an 8-bin hue histogram (cheque tints); the share of pixels on long horizontal
and vertical lines (morphological opening with long thin kernels: table rules and boxes); edge
density; a 24x24 grey thumbnail; ink profiles down (32) and across (16) the page.
`synthetic_page` renders a page of each class, photographs it and flattens it back with the true
homography, so training pages look like the pipeline's flat pages. `train` fits gradient-boosted
trees (150 pages per class, ~2 minutes) and validates on 50 other pages per class.
`classify_stage` sets the type when the model is at least 60% sure.

### 3.7 `findoc/export/excel.py`, CLI, pipeline, types

- `write_workbook`: four sheets, header bold and frozen, numbers as numbers, widths fitted.
- CLI: `findoc synth --kind invoice`, `findoc classify train`, `findoc invoice train`,
  `findoc export`; `process` prints invoice fields and a line-items table.
- Pipeline: `... enhance -> classify -> align ... -> ocr -> invoice -> risk`.
- `Document.line_items`; settings `FINDOC_DOCTYPE_MODEL`, `FINDOC_INVOICE_MODEL`,
  `FINDOC_INVOICE_EXTRACTOR`.

### 3.8 Results (`experiments/results/phase7.json`)

**Extraction** (90 synthetic invoices, 30 per layout; simulated OCR; laptop run, RTX 3050):

| OCR character errors | Extractor | Header precision | Recall | **F1** | Line-item cells | Time |
| --- | --- | --- | --- | --- | --- | --- |
| 0% | rules | 1.000 | 1.000 | **1.000** | 1.000 | 1.5 ms |
| 0% | LayoutLMv3 | 1.000 | 1.000 | **1.000** | 1.000 | 51 ms |
| 1% | rules | 0.953 | 0.893 | **0.922** | 0.878 | 1.3 ms |
| 1% | LayoutLMv3 | 0.955 | 0.937 | **0.946** | 0.981 | 49 ms |
| 3% | rules | 0.880 | 0.770 | **0.821** | 0.630 | 1.3 ms |
| 3% | LayoutLMv3 | 0.887 | 0.840 | **0.863** | 0.929 | 49 ms |

The SRS target is a field F1 of at least 0.85. The rules meet it on clean or slightly noisy
text and miss it at 3% character errors, where one wrong character makes a whole field wrong
under exact matching. **LayoutLMv3 meets it at 3% too**, and the difference is largest on line
items (0.93 vs 0.63): the rules find columns by header position and lose a row when a number
is split or merged, while LayoutLMv3 labels each token from its text, position and the image
together, so a garbled token in the right place is still the amount.

**Training** (`models/invoice_layoutlmv3/training.json`, 1,500 synthetic invoices, fp16):
about 10.5 minutes per epoch. Validation F1 was 0.980 (line items 0.995) after epoch 1 and
exactly the same after epochs 2-5, while the training loss fell from 0.65 to 0.001. The
remaining errors are in the noisy validation text itself, which more training on clean labels
can't fix, so epochs 2-5 were wasted: the default is now 2.

**Checks** (40 invoices per kind; "review" = the risk score's band is medium or high):

| OCR errors | Clean invoices sent to review | Wrong total | Wrong line | Wrong tax type | Invalid GSTIN |
| --- | --- | --- | --- | --- | --- |
| 0% | **0%** | 100% | 100% | 100% | 100% flagged (90% to review) |
| 1% | 15% | 98% | 88% | 90% | 92% flagged (90% to review) |
| 3% | 35% | 68% | 78% | 85% | 90% flagged (85% to review) |

The invalid GSTINs that don't reach review get only a low-confidence flag: the forged
character happens to be one known OCR confusion away from a valid number, so the check can't
tell forgery from misreading and says so in the reason. Before v0.8.2 this happened to a third
of them (62% to review at 0%), because the confusion table was too generous.

**Real OCR** (`--real-ocr`: 30 photographed invoices, EasyOCR, whole pipeline). Laptop run
with LayoutLMv3 (v0.8.1): header F1 **0.868**, line-item cells **0.951**, and `INVOICE_MATH`
raised no false alarm on a clean invoice while catching every wrong total and wrong line. But
`GSTIN_INVALID` fired on 13 of the 15 clean invoices. Reading the GSTINs back against the truth
showed they were misreads, and systematic ones:

| What EasyOCR printed | What was on the paper | Seen |
| --- | --- | --- |
| `...1Z2` / `...I22Z` | `...1Z7` / `...I2ZZ` (14th character is always Z) | 4 |
| `O` | `Q` (or `D`) in the PAN | 6 |
| `0` | `9` | 3 |
| `I` | `1` (entity number) | 2 |
| `Q` in a digit place | `9` (v0.8.3: 8 of 8 on the second laptop run) | 8 |
| `@` | a leading `0` (v0.8.3) | 3 |

v0.8.2 answers this in three layers (repair by position, a second read of the crop at 2-3x
with only 0-9/A-Z allowed, and a confusion table built from this list). Rerun on CPU with the
rules extractor (`phase7_real_ocr_rules.json`): GSTINs read exactly 73%, and **no clean invoice
got a full GSTIN alarm** (3 of 15 got the low-confidence "probably misread" flag, which doesn't
send them to review). On 20 fresh photos not used to build the table: 6 misreads, 5 correctly
explained (the suggested fix was the true number every time), 1 clean invoice with a full alarm
(two errors in one number). The rules extractor's line items fall to 0.49 on real OCR, and its
arithmetic alarms send 40% of clean invoices to review, so use LayoutLMv3 for real photos.

**On the laptop with LayoutLMv3** (`phase7_real_ocr.json` is the v0.8.3 run; ~3 s per invoice
on the GPU):

| | v0.8.1 | v0.8.2 | **v0.8.3** |
| --- | --- | --- | --- |
| Header field F1 / line-item cells | 0.868 / 0.951 | 0.928 / 0.951 | **0.940 / 0.951** |
| GSTINs read exactly | | 78% | **85%** |
| Clean invoices with a GSTIN alarm (any confidence) | 13 of 15 | 4 of 15 | 3 of 15 |
| Clean invoices sent to review | | 2 of 15 | **1 of 15 (7%)** |
| Planted errors sent to review by the right check | | 14 of 15 | **15 of 15 (100%)** |

The one clean invoice left (invoice 0) has two misread characters in one GSTIN; telling that
from a forged number would need a second confusion, which would excuse many more forgeries.
Better OCR, or the reviewer, is the right answer there.

**Classifier**: 100% on 200 fresh synthetic pages (50 per class). The synthetic classes look very
different, so this shows the features work, not real-world accuracy; real photos of other
documents will be harder.

#### Lessons worth telling in a viva

1. **Measure the checks under OCR noise, not on perfect text.** With exact text every planted
   error was caught and no clean invoice was flagged. At 1% character errors, 40% of clean
   invoices were flagged at first: one misread digit breaks the arithmetic.
2. **Use the document's own redundancy.** Three changes brought flags on clean invoices down
   to 20%, and reviews (medium or high risk) to 15%:
   - recognising a decimal comma (`,dd` at the end is never Indian grouping);
   - treating a disagreement that the rest of the invoice resolves as a misread, using the
     amount in words as tie-breaker;
   - only drawing conclusions from numbers that were actually read.
3. **Not every remaining error can be told from fraud by arithmetic.** A line with two misread
   digits looks like a changed amount. OCR confidence per word would help, which the real-OCR
   run measures.
4. **Simulated OCR errors aren't real OCR errors.** The simulator confuses characters at
   random; EasyOCR makes the same few mistakes again and again (O for Q, "1Z" as "12"). The
   GSTIN check looked fine on simulated text and failed on 13 of 15 real photos. Look at the
   actual misreads before designing the fix, and keep the "misread" excuse as narrow as the
   evidence: every confusion you allow is also a way for a forgery to pass.
5. **More epochs aren't free.** Log the validation score every epoch; here it stopped moving
   after the first, and four more epochs cost 42 minutes.
6. **Code you can't run here still needs a test.** The tiny offline LayoutLMv3 exercises the full
   training path, so the GPU run should only need the weights.

### 3.9 Tests (`tests/test_invoice.py`)

23 tests (the LayoutLMv3 one is skipped when transformers isn't installed, as in CI):
- GSTIN examples and OCR repair, including the position repairs and the second look (with a
  stand-in OCR engine);
- every layout, and each planted error breaking exactly what it should;
- dataset generation and receipt shape;
- normalisation, phrase splitting, and assembling the true labels back into exactly the truth;
- rule extraction on clean OCR (F1 and line items at least 0.95 per layout);
- each planted error caught by the right check, plus the misread edge cases;
- the invoice stage, the classifier (train, load, stage), the Excel export (numbers stay
  numbers), the synth CLI, and tiny LayoutLMv3 training plus extraction.

---

## 4. Concepts used in this phase

- **Document understanding**: key-value extraction, table structure from word geometry,
  reading order.
- **Transformers for documents**: token, 2D-position and image-patch embeddings; self-attention;
  pre-training then fine-tuning.
- **Token classification with BIO tags**; sub-word tokenisation (byte-level BPE) and aligning
  labels to the first sub-word.
- **Check digits** (Luhn-style, mod 36) and error-detecting codes.
- **Redundancy and error correction**: deriving each quantity two ways to localise a single
  error, the same idea as parity and checksums.
- **Precision, recall, F1** at field level; exact-match scoring.
- **Gradient-boosted trees on hand-designed features** for classification; synthetic
  training data.
- **GST basics**: GSTIN, HSN codes, CGST/SGST vs. IGST, Indian financial year.

---

## 5. Try it yourself (and the GPU steps)

1. `findoc synth --kind invoice --n 5` and open the photos and `labels.json`.
2. `findoc classify train` (~2 min), then `findoc process` one of the invoices with OCR: which
   fields and checks come out? Export with `findoc export`.
3. **GPU**: `pip install -e ".[layout]"` then `findoc invoice train` (downloads layoutlmv3-base
   once; 2 epochs, ~21 min on an RTX 3050). Then
   `python experiments/phase7_invoice_eval.py --layout models/invoice_layoutlmv3` to compare it
   with the rules at 0/1/3% character errors.
4. **Real OCR**: `python experiments/phase7_invoice_eval.py --real-ocr --n 30` runs photographed
   invoices through EasyOCR and the whole pipeline. The JSON has one row per invoice: each
   GSTIN and the total as read and as printed, and every check with its confidence. Find an
   invoice sent to review and explain why from its row.
5. Change one digit of a printed total in an invoice photo and process it: which check fires,
   and does the reason mention the amount in words?
6. In `invoice_rules.py`, turn off the misread handling (report every disagreement) and re-run
   the checks part of the experiment at 1%: how many clean invoices go to review?

---

## 6. Viva and interview questions

**Q: Why not use templates for invoices, like cheques?**
Every vendor prints its own layout; there is no small set of templates. Fields must be found from
text and position.

**Q: How does LayoutLMv3 differ from running a text NER model on the OCR text?**
It also sees each word's box and the page image, so layout (columns, alignment, headers) and
appearance (bold totals, table lines) inform the label, not only the words.

**Q: Why BIO tags, and why label only the first sub-word?**
BIO separates adjacent fields of the same type and multi-word values. A word split into several
tokens should count once; the extra tokens get -100 so the loss ignores them.

**Q: How do you validate a GSTIN?**
Format, state code and the mod-36 check character; a misread character usually breaks the
check, so a single OCR confusion that would make it valid is reported as a probable misread.

**Q: What does `INVOICE_MATH` verify, and how do you avoid blaming OCR errors on the vendor?**
Lines, subtotal, tax rate, CGST = SGST, total. The invoice states most numbers twice; a number
that disagrees while its other derivation fits everything else is treated as a misread. On
simulated OCR with 1% character errors, clean invoices flagged fell from 40% to 20%, and those
sent to review to 15%.

**Q: When is IGST charged instead of CGST + SGST?**
When supplier and buyer are in different states (the GSTIN's first two digits differ).

**Q: Your classifier scored 100%. Is that the real accuracy?**
No: synthetic classes are very different. It shows the features work; real documents of
other kinds will be harder, which is why the template match still decides cheques and text cues
back it up.

**Q: How would you improve extraction under OCR noise?**
Use OCR confidence per word in the checks, a fuzzy match for names and numbers, LayoutLMv3
trained with more noise, and a field-level "did it parse" prior learned from data.
