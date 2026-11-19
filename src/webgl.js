import { ReferenceRenderer, loadImages } from "./reference.js";

export { ReferenceRenderer as LockedWebGLRenderer, loadImages };

export async function loadRenderer(canvas, meta) {
  const images = await loadImages(meta);
  return new ReferenceRenderer(canvas, meta, images);
}
