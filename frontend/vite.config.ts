import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the API runs on :8000; proxying keeps the frontend code
// origin-relative ("/api/...") so the same build works behind any host.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: "0.0.0.0",
    proxy: { "/api": process.env.VITE_DEV_API ?? "http://localhost:8000" }
  },
  build: {
    chunkSizeWarningLimit: 900
  },
  // Pre-bundle these at server start instead of discovering them on the first
  // page load, which otherwise forces a "new dependencies optimized, reloading"
  // mid-load reload every time the dev server starts cold.
  optimizeDeps: {
    include: ["react", "react-dom", "react-dom/client", "react-router-dom", "recharts", "clsx", "lucide-react"]
  }
});
