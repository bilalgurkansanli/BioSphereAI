import type { Metadata } from "next";
import { PropertiesView } from "@/components/dashboard/PropertiesView";

export const metadata: Metadata = { title: "Evler" };

export default function PropertiesPage() {
  return <PropertiesView />;
}
