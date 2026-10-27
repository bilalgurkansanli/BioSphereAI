import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

/**
 * Builder C's modules (`lib/capture/{coverage,plan,detectors}.ts`) are deliberately free of
 * React, `three` and the DOM, so they run under plain Node with no environment shim and no
 * transform beyond esbuild's TypeScript strip. The only thing vitest needs told is the
 * `@/*` path alias, which lives in tsconfig.json and which vite does not read by itself.
 *
 * `globals` is left OFF on purpose: the tests import `describe`/`it`/`expect` explicitly, so
 * `npx tsc --noEmit` and `next build` type-check them without any ambient type registration.
 */
export default defineConfig({
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
