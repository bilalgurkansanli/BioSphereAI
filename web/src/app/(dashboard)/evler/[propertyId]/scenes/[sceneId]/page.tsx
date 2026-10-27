import { SceneEditor } from "@/components/dashboard/SceneEditor";

/** Next 16: dynamic params arrive as a Promise and must be awaited. */
export default async function SceneEditorPage({
  params,
}: {
  params: Promise<{ propertyId: string; sceneId: string }>;
}) {
  const { propertyId, sceneId } = await params;
  return <SceneEditor propertyId={propertyId} sceneId={sceneId} />;
}
