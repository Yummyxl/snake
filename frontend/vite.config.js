import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: process.env.FRONTEND_HOST ?? "0.0.0.0",
    port: Number(process.env.FRONTEND_PORT ?? 5173),
    proxy: {
      "/api": {
        target: process.env.VITE_PROXY_TARGET ?? `http://127.0.0.1:${process.env.BACKEND_PORT ?? 8000}`,
        changeOrigin: true,
      },
    },
  },
  preview: {
    host: process.env.FRONTEND_HOST ?? "0.0.0.0",
    port: Number(process.env.FRONTEND_PREVIEW_PORT ?? 4173),
  },
});
