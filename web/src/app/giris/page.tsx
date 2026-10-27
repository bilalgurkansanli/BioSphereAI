import type { Metadata } from "next";
import { Suspense } from "react";
import { SessionProvider } from "@/components/auth/SessionProvider";
import { SignInView } from "@/components/auth/SignInView";

export const metadata: Metadata = {
  title: "Giriş",
  description: "BioSphere AI hesabınıza girin veya yeni bir hesap oluşturun.",
  // A sign-in page has nothing for a search engine and every reason not to be a landing
  // page for one.
  robots: { index: false, follow: false },
};

/**
 * `useSearchParams` opts a page into client rendering, and Next requires the boundary to be
 * explicit — without the Suspense wrapper the build fails rather than warns. The fallback is
 * blank on purpose: this page paints in milliseconds and a spinner would only flash.
 */
export default function SignInPage() {
  return (
    <SessionProvider>
      <Suspense fallback={<div className="min-h-dvh" />}>
        <SignInView />
      </Suspense>
    </SessionProvider>
  );
}
