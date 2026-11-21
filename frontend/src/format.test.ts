import { describe, expect, it } from "vitest";
import { ago, bytes, checkLabel, confidenceOf, fieldLabel, fieldsForCheck, rupees } from "./format";

describe("format", () => {
  it("labels checks and fields", () => {
    expect(checkLabel("AMOUNT_MISMATCH")).toBe("Amount mismatch");
    expect(fieldLabel("vendor_gstin")).toBe("Vendor GSTIN");
    expect(fieldLabel("some_new_field")).toBe("Some new field");
  });

  it("links checks to the fields they are about", () => {
    expect(fieldsForCheck("DATE_STALE", {})).toEqual(["date"]);
    expect(fieldsForCheck("GSTIN_INVALID", { party: "buyer" })).toEqual(["buyer_gstin"]);
    expect(fieldsForCheck("TAMPER_REGION", { top: "payee" })).toEqual(["payee"]);
    expect(fieldsForCheck("TAMPER_REGION", { field_scores: { payee: 0.9, date: 0.1 } })).toEqual(["payee"]);
    expect(fieldsForCheck("DUPLICATE", {})).toEqual([]);
  });

  it("formats amounts the Indian way", () => {
    expect(rupees("1234567.5")).toBe("12,34,567.50");
    expect(rupees("abc")).toBe("abc");
    expect(rupees(null)).toBeNull();
  });

  it("formats times and sizes", () => {
    const now = new Date("2026-11-21T12:00:00Z");
    expect(ago("2026-11-21T11:59:40Z", now)).toBe("just now");
    expect(ago("2026-11-21T11:30:00Z", now)).toBe("30 minutes ago");
    expect(ago("2026-11-20T12:00:00Z", now)).toBe("yesterday");
    expect(ago("2026-11-21T11:00:00", now)).toBe("1 hour ago"); // naive = UTC
    expect(bytes(2048)).toBe("2 KB");
    expect(bytes(5 * 1024 ** 2)).toBe("5.0 MB");
    expect(confidenceOf(0.95)).toBe("high");
    expect(confidenceOf(0.4)).toBe("low");
  });
});
