import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"], display: "swap" });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"], display: "swap" });

const DESCRIPTION =
  "Telefonla çekilen bir oda videosundan 8K'ya kadar 360° panorama, odaları birbirine bağlayan bir oynatıcı, kat planı ve deneysel bir 3B gezinti. MIT lisanslı, kendi bilgisayarınızda çalışan açık kaynak bir mühendislik projesi.";

export const metadata: Metadata = {
  title: {
    default: "BioSphere AI — Telefon videosundan 360° sanal tur · açık kaynak",
    template: "%s · BioSphere AI",
  },
  description: DESCRIPTION,
  applicationName: "BioSphere AI",
  authors: [{ name: "bilalgurkansanli", url: "https://github.com/bilalgurkansanli" }],
  keywords: ["360", "panorama", "virtual tour", "sanal tur", "open source", "açık kaynak", "stitching", "three.js"],
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000"),
  openGraph: {
    title: "BioSphere AI — telefon videosundan 360° sanal tur",
    description:
      "Açık kaynak (MIT): telefon videosundan 8K'ya kadar 360° panorama, oynatıcı, kat planı. Kendi bilgisayarınızda çalışır.",
    type: "website",
    siteName: "BioSphere AI",
    locale: "tr_TR",
    alternateLocale: ["en_GB"],
  },
  twitter: {
    card: "summary_large_image",
    title: "BioSphere AI — telefon videosundan 360° sanal tur",
    description: "Açık kaynak (MIT): telefon videosundan 8K'ya kadar 360° panorama, oynatıcı ve kat planı.",
  },
};

export const viewport: Viewport = {
  themeColor: "#05070a",
  colorScheme: "dark",
  width: "device-width",
  initialScale: 1,
  // The player uses the full viewport on phones, including under the notch.
  viewportFit: "cover",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="tr" suppressHydrationWarning>
      {/* Extensions (password managers, Colorzilla, Grammarly) inject attributes onto
          <body> before React hydrates, which React reports as a mismatch. Suppressing it
          here is the standard remedy; it only affects this element's own attributes, not
          the tree below it. */}
      <body
        suppressHydrationWarning
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
