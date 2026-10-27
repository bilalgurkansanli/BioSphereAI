import { PropertyDetailView } from "@/components/dashboard/PropertyDetailView";

/** Next 16: dynamic params arrive as a Promise and must be awaited. */
export default async function PropertyDetailPage({ params }: { params: Promise<{ propertyId: string }> }) {
  const { propertyId } = await params;
  return <PropertyDetailView propertyId={propertyId} />;
}
