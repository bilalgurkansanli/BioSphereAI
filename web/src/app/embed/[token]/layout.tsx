/**
 * Chrome-less shell for the public embed.
 *
 * This is a nested layout, so it cannot render <html> or <body> — the root layout owns
 * those. What it can do is pin itself over the viewport and neutralise the document
 * scrolling that the dashboard's styles assume, which is enough: `position: fixed` means
 * the player never depends on the height of anything above it in the tree.
 */

export const dynamic = "force-dynamic";

export default function EmbedLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      {/* `scrollbar-gutter: stable` is right for the dashboard and wrong inside an iframe,
          where it reserves a visible strip beside a full-bleed panorama. */}
      <style>{`
        html { overflow: hidden; scrollbar-gutter: auto; color-scheme: dark; }
        body { overflow: hidden; margin: 0; padding: 0; min-height: 100dvh; background: var(--canvas); }
      `}</style>
      <div data-theme="dark" className="fixed inset-0 overflow-hidden bg-canvas">
        {children}
      </div>
    </>
  );
}
