import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the UI on :5173 and forwards API calls to `findoc api serve` on :8000.
const api = { target: "http://127.0.0.1:8000", changeOrigin: false };

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": api, "/health": api, "/metrics": api, "/docs": api, "/openapi.json": api } },
  build: { outDir: "dist", sourcemap: false },
  test: { environment: "node" },
});
