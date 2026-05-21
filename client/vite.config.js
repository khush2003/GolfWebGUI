import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const BACKEND = process.env.NEUROGOLF_BACKEND || "http://127.0.0.1:8081";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 3000,
    strictPort: false,
    proxy: {
      "/api": { target: BACKEND, changeOrigin: true },
    },
  },
});
