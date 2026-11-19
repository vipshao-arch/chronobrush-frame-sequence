import { defineConfig } from "vite";

export default defineConfig({
  // Relative asset paths keep the built site working under GitHub Pages.
  base: "./",
  build: {
    outDir: "docs",
    emptyOutDir: true,
  },
});
