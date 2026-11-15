const CACHE_NAME = "steelpipe-runtime-v1";
const WASM_SUFFIX = "/vendor/onnxruntime/ort-wasm-simd-threaded.wasm";

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

async function serveCompressedWasm(request) {
  if (typeof DecompressionStream === "undefined") {
    return fetch(request);
  }

  const compressedUrl = new URL(request.url);
  compressedUrl.pathname += ".gz";
  const cache = await caches.open(CACHE_NAME);
  let compressed = await cache.match(compressedUrl.href);
  if (!compressed) {
    compressed = await fetch(compressedUrl.href, { cache: "force-cache" });
    if (!compressed.ok || !compressed.body) {
      return fetch(request);
    }
    await cache.put(compressedUrl.href, compressed.clone());
  }

  const decompressed = compressed.body.pipeThrough(
    new DecompressionStream("gzip"),
  );
  self.clients.matchAll({ type: "window", includeUncontrolled: true }).then(
    (clients) => {
      for (const client of clients) {
        client.postMessage({ type: "compressed-wasm-served" });
      }
    },
  );
  return new Response(decompressed, {
    status: 200,
    headers: {
      "Content-Type": "application/wasm",
      "Cross-Origin-Resource-Policy": "same-origin",
      "Cache-Control": "public, max-age=31536000, immutable",
    },
  });
}

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (
    url.origin === self.location.origin &&
    url.pathname.endsWith(WASM_SUFFIX)
  ) {
    event.respondWith(serveCompressedWasm(event.request));
  }
});
