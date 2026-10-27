import type { Metadata } from "next";
import { WalkPageView } from "@/components/dashboard/WalkPageView";

export const metadata: Metadata = { title: "3B Gezinti" };

export default async function WalkPage({ params }: { params: Promise<{ propertyId: string }> }) {
  const { propertyId } = await params;
  return <WalkPageView propertyId={propertyId} />;
}
