/**
 * The assistant's answer as formatted text: paragraphs, bold labels, bullet lists, links.
 *
 * ── WHY NOT A MARKDOWN LIBRARY ────────────────────────────────────────────────────────
 * The prompt asks for exactly four things — bold, lists, links, paragraphs — and this renders
 * exactly those, as React nodes. Never `dangerouslySetInnerHTML`: the text is a model's output
 * that quotes web pages, and the one property a renderer here must have is that nothing in the
 * answer can become markup. A raw `<script>` in an answer is shown as the characters it is.
 *
 * ── LINKS ARE http(s) ONLY, AND OPEN ELSEWHERE ─────────────────────────────────────────
 * `javascript:` and `data:` URLs are text, not links. Every link opens in a new tab with
 * `noopener noreferrer nofollow`: the player is framed on someone else's listing page, and a
 * source the model cited must neither navigate that page away nor hand it a referrer.
 *
 * ── IT IS RENDERED WHILE IT STREAMS ───────────────────────────────────────────────────
 * Half an answer is valid input. An unclosed `**` is shown as written until its partner
 * arrives, and a half-typed link stays text until it is complete.
 *
 * The parsing lives in `markdown.ts`, free of React, so the tests run it under plain Node.
 */

import { Fragment, type ReactNode } from "react";
import { cn } from "@/lib/utils";
import { parseBlocks, tokenizeInline } from "./markdown";

function renderInline(text: string): ReactNode[] {
  return tokenizeInline(text).map((token, index) => {
    switch (token.kind) {
      case "bold":
        return (
          <strong key={index} className="font-semibold text-ink">
            {token.text}
          </strong>
        );
      case "italic":
        return <em key={index}>{token.text}</em>;
      case "link":
        return (
          <a
            key={index}
            href={token.href}
            target="_blank"
            rel="noopener noreferrer nofollow"
            className="text-sky underline decoration-sky/40 underline-offset-2 transition-colors hover:decoration-sky"
          >
            {token.text}
          </a>
        );
      default:
        return <Fragment key={index}>{token.text}</Fragment>;
    }
  });
}

export function MarkdownLite({ text, className }: { text: string; className?: string }) {
  return (
    <div className={cn("space-y-2 text-[12.5px] leading-relaxed text-ink-muted", className)}>
      {parseBlocks(text).map((block, index) => {
        if (block.kind === "label") {
          return (
            <p key={index} className="pt-1 text-[12px] font-semibold tracking-tight text-ink">
              {renderInline(block.text)}
            </p>
          );
        }
        if (block.kind === "paragraph") {
          return (
            <p key={index}>
              {block.lines.map((line, i) => (
                <Fragment key={i}>
                  {i > 0 ? <br /> : null}
                  {renderInline(line)}
                </Fragment>
              ))}
            </p>
          );
        }
        const ListTag = block.ordered ? "ol" : "ul";
        return (
          <ListTag
            key={index}
            className={cn("space-y-1 pl-4", block.ordered ? "list-decimal" : "list-disc marker:text-ink-faint")}
          >
            {block.items.map((item, i) => (
              <li key={i}>
                {renderInline(item.text)}
                {item.children.length ? (
                  <ul className="mt-1 list-[circle] space-y-0.5 pl-4 marker:text-ink-faint">
                    {item.children.map((child, j) => (
                      <li key={j}>{renderInline(child)}</li>
                    ))}
                  </ul>
                ) : null}
              </li>
            ))}
          </ListTag>
        );
      })}
    </div>
  );
}
