import type { Metadata } from "next";
import { StagingStudio } from "@/components/restyle/StagingStudio";

export const metadata: Metadata = { title: "Sanal Dekorasyon" };

export default function DecorPage() {
  return <StagingStudio />;
}
