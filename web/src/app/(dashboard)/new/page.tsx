import type { Metadata } from "next";
import { NewPropertyWizard } from "@/components/dashboard/NewPropertyWizard";

export const metadata: Metadata = { title: "Yeni Ev" };

export default function NewPropertyPage() {
  return <NewPropertyWizard />;
}
