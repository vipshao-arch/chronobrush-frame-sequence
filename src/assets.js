export function assetUrl(file) {
  return new URL(`assets/${file}`, document.baseURI).href;
}
