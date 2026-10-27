import { describe, expect, it } from "vitest";

import { sseFrames } from "@/lib/api";
import { fill, formatPct, formatRate, formatTl, parseTypedNumber } from "../format";
import { parseBlocks, tokenizeInline } from "../markdown";

describe("the answer's inline formatting", () => {
  it("reads bold, markdown links and bare links", () => {
    expect(tokenizeInline("Brüt getiri **%4,88** ([Emlakjet, Eylül 2026](https://www.emlakjet.com/a))")).toEqual([
      { kind: "text", text: "Brüt getiri " },
      { kind: "bold", text: "%4,88" },
      { kind: "text", text: " (" },
      { kind: "link", text: "Emlakjet, Eylül 2026", href: "https://www.emlakjet.com/a" },
      { kind: "text", text: ")" },
    ]);
    expect(tokenizeInline("kaynak: https://www.tcmb.gov.tr/x.pdf.")).toEqual([
      { kind: "text", text: "kaynak: " },
      { kind: "link", text: "tcmb.gov.tr", href: "https://www.tcmb.gov.tr/x.pdf" },
      { kind: "text", text: "." },
    ]);
  });

  it("never turns a script URL or HTML into anything but text", () => {
    const tokens = tokenizeInline('[tıkla](javascript:alert(1)) <img src=x onerror="alert(1)">');
    expect(tokens.every((token) => token.kind === "text")).toBe(true);
    expect(tokens.map((token) => token.text).join("")).toContain("<img src=x");
  });

  it("leaves an unclosed bold as it is while the answer is still streaming", () => {
    expect(tokenizeInline("Net getiri **%3,8")).toEqual([{ kind: "text", text: "Net getiri **%3,8" }]);
  });

  it("does not read a multiplication or a footnote star as italics", () => {
    expect(tokenizeInline("12 * 60.000 TL")).toEqual([{ kind: "text", text: "12 * 60.000 TL" }]);
  });
});

describe("the answer's blocks", () => {
  it("splits labels, paragraphs and lists, with one level of nesting", () => {
    const blocks = parseBlocks(
      [
        "Brüt kira getirisi %4,88.",
        "",
        "**İlan bilgileri**",
        "- Fiyat 14.750.000 TL",
        "- Net hesap şu varsayımlara dayanıyor:",
        "  - yılda 1 ay boş kalma;",
        "  - bakım gideri",
        "",
        "Sonuç kiraya bağlı.",
      ].join("\n"),
    );
    expect(blocks.map((block) => block.kind)).toEqual(["paragraph", "label", "list", "paragraph"]);
    const list = blocks[2];
    expect(list.kind === "list" && list.items[1].children).toEqual(["yılda 1 ay boş kalma;", "bakım gideri"]);
  });
});

describe("the event stream", () => {
  it("reads frames cut at any byte, and keeps the unfinished tail", () => {
    const body = 'data: {"type":"text","delta":"Brüt"}\n\n: ping\n\ndata: {"type":"done","turns_left":3,"full":false}\n\n';
    let buffer = "";
    const frames: string[] = [];
    for (let i = 0; i < body.length; i += 7) {
      buffer += body.slice(i, i + 7);
      const result = sseFrames(buffer);
      frames.push(...result.frames);
      buffer = result.rest;
    }
    expect(frames.map((frame) => JSON.parse(frame).type)).toEqual(["text", "done"]);
    expect(buffer).toBe("");
  });
});

describe("numbers for a buyer", () => {
  it("formats Turkish and English", () => {
    expect(formatTl(14_750_000, "tr")).toBe("14.750.000 TL");
    expect(formatTl(14_750_000, "en")).toBe("14,750,000 TL");
    expect(formatPct(4.8815, "tr")).toBe("%4,88");
    expect(formatPct(4.8815, "en")).toBe("4.88%");
    expect(formatRate(50, "tr")).toBe("%50");
    expect(formatRate(2.85, "tr")).toBe("%2,85");
    expect(fill("{down} peşinat · {term} ay", { down: "%50", term: 120 })).toBe("%50 peşinat · 120 ay");
  });

  it("reads what an agent types, and empty is not zero", () => {
    expect(parseTypedNumber("14.750.000", "money")).toBe(14_750_000);
    expect(parseTypedNumber("14 750 000", "money")).toBe(14_750_000);
    expect(parseTypedNumber("118,5", "decimal")).toBe(118.5);
    expect(parseTypedNumber("", "money")).toBeNull();
    expect(Number.isNaN(parseTypedNumber("on beş", "money"))).toBe(true);
  });
});
