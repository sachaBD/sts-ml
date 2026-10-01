// Shared browser-model plumbing. Architecture-specific models supply their own forward passes.
(() => {
  const b64 = (s) => new Float32Array(Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer);
  const dataRoot = () => ((typeof window !== "undefined" && window.GUI_ROOT) ? window.GUI_ROOT : "./") + "data/";
  const json = (name) => fetch(dataRoot() + name).then((r) => {
    if (!r.ok) throw new Error(`could not load model data: ${name}`);
    return r.json();
  });
  const api = { b64, dataRoot, json };
  if (typeof window !== "undefined") window.BrowserModel = api;
  if (typeof module !== "undefined") module.exports = api;
})();
