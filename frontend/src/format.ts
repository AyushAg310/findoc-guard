// Display helpers: plain functions so they can be unit-tested without a browser.

export const STATUS_LABEL: Record<string, string> = {
  queued: "Queued",
  processing: "Processing",
  needs_review: "Needs review",
  processed: "Processed",
  rejected_quality: "Retake needed",
  failed: "Failed",
  approved: "Approved",
  rejected: "Rejected",
  escalated: "Escalated",
};

export const FIELD_LABEL: Record<string, string> = {
  payee: "Payee",
  amount_figures: "Amount (figures)",
  amount_words_1: "Amount in words, line 1",
  amount_words_2: "Amount in words, line 2",
  amount_words: "Amount in words",
  date: "Date",
  account_number: "Account number",
  micr: "MICR line",
  ifsc: "IFSC",
  cheque_number: "Cheque number",
  vendor: "Vendor",
  vendor_gstin: "Vendor GSTIN",
  buyer: "Buyer",
  buyer_gstin: "Buyer GSTIN",
  invoice_no: "Invoice number",
  subtotal: "Subtotal",
  cgst: "CGST",
  sgst: "SGST",
  igst: "IGST",
  total: "Total",
};

export const STAGE_LABEL: Record<string, string> = {
  load: "Reading the file",
  video: "Reading the video",
  track: "Tracking the document",
  quality: "Checking image quality",
  detect: "Finding the page",
  recon3d: "3D reconstruction",
  rectify: "Flattening the page",
  enhance: "Cleaning up the image",
  classify: "Recognising the document type",
  align: "Matching the cheque template",
  fields: "Reading fields",
  rules: "Checking rules",
  tamper: "Looking for tampering",
  duplicate: "Looking for duplicates",
  signature: "Comparing the signature",
  ocr: "Reading the text",
  invoice: "Reading the invoice",
  risk: "Scoring risk",
};

/** Turns a check id like AMOUNT_MISMATCH into "Amount mismatch". */
export function checkLabel(id: string): string {
  const s = id.toLowerCase().replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function fieldLabel(name: string): string {
  return FIELD_LABEL[name] ?? checkLabel(name);
}

/** Which fields on the page a fired check is about, so the canvas can highlight them. */
export function fieldsForCheck(id: string, evidence: Record<string, unknown>): string[] {
  if (id === "AMOUNT_MISMATCH") return ["amount_figures", "amount_words_1", "amount_words_2", "amount_words"];
  if (id.startsWith("DATE_")) return ["date"];
  if (id.startsWith("SIGNATURE")) return ["signature"];
  if (id === "GSTIN_INVALID") return [`${String(evidence.party ?? "vendor")}_gstin`];
  if (id === "TAMPER_REGION") {
    const top = evidence.top;
    if (Array.isArray(top)) return top.map((t) => (Array.isArray(t) ? String(t[0]) : String(t)));
    if (typeof top === "string") return [top];
    const scores = evidence.field_scores as Record<string, number> | undefined;
    if (scores) return Object.entries(scores).filter(([, v]) => v >= 0.5).map(([k]) => k);
  }
  if (id === "INVOICE_MATH") return ["subtotal", "cgst", "sgst", "igst", "total"];
  return [];
}

export type Confidence = "high" | "low";
export const confidenceOf = (c: number): Confidence => (c >= 0.8 ? "high" : "low");

const rel = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

/** "3 minutes ago", "yesterday", or a date for anything older than a week. */
export function ago(iso: string, now: Date = new Date()): string {
  const then = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z");
  const s = (then.getTime() - now.getTime()) / 1000;
  const abs = Math.abs(s);
  if (abs < 45) return "just now";
  if (abs < 3600) return rel.format(Math.round(s / 60), "minute");
  if (abs < 86400) return rel.format(Math.round(s / 3600), "hour");
  if (abs < 7 * 86400) return rel.format(Math.round(s / 86400), "day");
  return then.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}

/** Indian digit grouping for amounts: 1234567.5 -> "12,34,567.50". */
export function rupees(value: string | null): string | null {
  if (value == null || !/^-?\d+(\.\d+)?$/.test(value)) return value;
  return Number(value).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export const MONEY_FIELDS = new Set(["amount_figures", "subtotal", "cgst", "sgst", "igst", "total"]);
