/**
 * Файл конфигурации Vite.
 *
 * Содержит:
 * - defineConfig: настройка React-плагина и dev proxy для backend API.
 */
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/app/",
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5174,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
