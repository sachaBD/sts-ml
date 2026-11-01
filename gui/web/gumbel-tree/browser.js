// The local API and the static browser backend return the same decision JSON.
export function browserAPI() {
  const worker = new Worker(new URL("worker.js", import.meta.url));
  const pending = new Map();
  let next = 0;
  worker.onmessage = ({ data: { id, result, error } }) => {
    const call = pending.get(id);
    if (!call) return;
    pending.delete(id);
    if (error) call.reject(new Error(error)); else call.resolve(result);
  };
  worker.onerror = (event) => {
    for (const call of pending.values()) call.reject(new Error(event.message || "browser worker failed"));
    pending.clear();
  };
  return (path, body) => new Promise((resolve, reject) => {
    const id = next++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, path, body });
  });
}
