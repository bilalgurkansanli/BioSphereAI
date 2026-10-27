/**
 * The parsing half of `MarkdownLite`, kept free of React so it runs under the plain-Node test
 * environment: lines into blocks, a line into inline tokens. See `MarkdownLite.tsx` for why the
 * grammar is this small and why nothing here can ever become markup.
 */

export type InlineToken =
  | { kind: "text"; text: string }
  | { kind: "bold"; text: string }
  | { kind: "italic"; text: string }
  | { kind: "link"; text: string; href: string };

export type Block =
  | { kind: "label"; text: string }
  | { kind: "paragraph"; lines: string[] }
  | { kind: "list"; ordered: boolean; items: { text: string; children: string[] }[] };

/** Bold, a markdown link (http/https only), a bare http(s) URL, italics — in that order. */
const INLINE =
  /\*\*([^*\n]+?)\*\*|\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)|(https?:\/\/[^\s<>()]+[^\s<>().,;:!?'"])|(?<![\w*])\*([^*\s][^*\n]*?)\*(?![\w*])/g;

export function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

export function tokenizeInline(text: string): InlineToken[] {
  const out: InlineToken[] = [];
  let last = 0;
  for (const match of text.matchAll(INLINE)) {
    const index = match.index ?? 0;
    if (index > last) out.push({ kind: "text", text: text.slice(last, index) });
    if (match[1] !== undefined) out.push({ kind: "bold", text: match[1] });
    else if (match[2] !== undefined && match[3] !== undefined) out.push({ kind: "link", text: match[2], href: match[3] });
    else if (match[4] !== undefined) out.push({ kind: "link", text: hostOf(match[4]), href: match[4] });
    else if (match[5] !== undefined) out.push({ kind: "italic", text: match[5] });
    last = index + match[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

const BULLET = /^(\s*)(?:[-*•]|\d+[.)])\s+(.*)$/;
const LABEL = /^\s*(?:#{1,6}\s+(.+?)|\*\*([^*]+)\*\*:?)\s*$/;

/** Lines into blocks: labels, paragraphs and (one level of nested) lists. */
export function parseBlocks(source: string): Block[] {
  const blocks: Block[] = [];
  let paragraph: string[] = [];
  let list: Extract<Block, { kind: "list" }> | null = null;

  const flushParagraph = () => {
    if (paragraph.length) blocks.push({ kind: "paragraph", lines: paragraph });
    paragraph = [];
  };
  const flushList = () => {
    if (list) blocks.push(list);
    list = null;
  };

  for (const raw of source.replace(/\r\n/g, "\n").split("\n")) {
    const line = raw.replace(/\s+$/, "");
    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }
    const bullet = BULLET.exec(line);
    if (bullet) {
      flushParagraph();
      const current = list as Extract<Block, { kind: "list" }> | null;
      if (bullet[1].length >= 2 && current && current.items.length > 0) {
        current.items[current.items.length - 1].children.push(bullet[2]);
        continue;
      }
      const ordered = /^\s*\d/.test(line);
      if (!current || current.ordered !== ordered) {
        flushList();
        list = { kind: "list", ordered, items: [] };
      }
      (list as Extract<Block, { kind: "list" }>).items.push({ text: bullet[2], children: [] });
      continue;
    }
    const label = LABEL.exec(line);
    if (label) {
      flushParagraph();
      flushList();
      blocks.push({ kind: "label", text: (label[1] ?? label[2] ?? "").trim() });
      continue;
    }
    flushList();
    paragraph.push(line.trim());
  }
  flushParagraph();
  flushList();
  return blocks;
}
