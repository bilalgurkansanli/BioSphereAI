import type { ReactNode } from "react";
import { SiteFooter } from "@/components/site/SiteFooter";
import { SiteHeader } from "@/components/site/SiteHeader";

/**
 * The public face of the open-source project: one page, its header and its footer.
 *
 * A route group, so none of this appears in the URL. The dashboard has its own group and its
 * own chrome; the two never share a shell. There are no legal pages here on purpose — there
 * is no company and no service; whoever runs a public instance writes their own notice.
 */
export default function SiteLayout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-dvh flex-col">
      <SiteHeader />
      <main className="flex-1">{children}</main>
      <SiteFooter />
    </div>
  );
}
