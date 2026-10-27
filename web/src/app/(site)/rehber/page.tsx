import type { Metadata } from "next";
import { Guide } from "@/components/site/Guide";

export const metadata: Metadata = {
  title: "Rehber — BioSphere'i kendi bilgisayarınızda çalıştırın",
  description: "BioSphere'i Windows, Linux ya da macOS'ta adım adım kurun ve ilk 360° turunuzu oluşturun.",
};

/** The step-by-step guide: BioSphere is software you run yourself, not a hosted service. */
export default function GuidePage() {
  return <Guide />;
}
