const test = require("node:test");
const assert = require("node:assert/strict");
const { createHash } = require("node:crypto");
const { copyFileSync, mkdtempSync, rmSync, writeFileSync } = require("node:fs");
const { tmpdir } = require("node:os");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

test("the model survives reload and recovers from a partial cached download", async t => {
  const directory = mkdtempSync(path.join(tmpdir(), "pulp-cache-test-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const base = pathToFileURL(directory + path.sep).href;
  const bytes = Buffer.from("GGUF-test-model-weights");
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  const file = "student-" + sha256.slice(0, 16) + ".gguf";
  const url = base + file;
  const manifest = { file, bytes: bytes.length, sha256 };
  const store = new Map();
  let modelRequests = 0;

  t.mock.method(globalThis, "fetch", async address => {
    if (String(address) === base + "model.json") return Response.json(manifest);
    assert.equal(String(address), url);
    modelRequests++;
    // The Fetch API provides decoded bytes while the HTTP header describes the encoded transfer.
    return new Response(bytes, { headers: { "Content-Length": "9", "Content-Encoding": "gzip" } });
  });

  class CacheManager {
    async getNameFromURL(address) {
      assert.equal(address, url);
      return "storage-key-" + file;
    }
    async open(key) { return store.get(key)?.blob ?? null; }
    async delete(key) { store.delete(key); }
    async download(address) {
      const response = await fetch(address);
      store.set(await this.getNameFromURL(address), {
        blob: await response.blob(),
        originalSize: Number(response.headers.get("Content-Length")),
      });
    }
  }

  class Wllama {
    constructor() { this.cacheManager = new CacheManager(); }
    // Reproduce the old runtime URL loader's transfer-length check, including its erroneous warm-cache miss.
    async loadModelFromUrl(address) {
      const key = await this.cacheManager.getNameFromURL(address);
      const entry = store.get(key);
      if (!entry || entry.blob.size !== entry.originalSize) await this.cacheManager.download(address);
      return this.loadModel([await this.cacheManager.open(key)]);
    }
    async loadModel(blobs) {
      assert.equal(blobs.length, 1);
      assert.deepEqual(Buffer.from(await blobs[0].arrayBuffer()), bytes);
    }
    async createCompletion() { return { choices: [{ text: "loaded" }] }; }
  }

  // Import a local runtime double through the real loader's normal dynamic import.
  const globals = {
    window: globalThis,
    document: { currentScript: { src: base + "llm.js" } },
    navigator: { locks: { request: async (key, callback) => { assert.equal(key, url); return callback(); } } },
    pulpTestRuntime: { Wllama, CacheManager },
  };
  for (const [name, value] of Object.entries(globals)) {
    const original = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { value, configurable: true });
    t.after(() => original ? Object.defineProperty(globalThis, name, original) : delete globalThis[name]);
  }
  writeFileSync(path.join(directory, "wllama.js"),
    "exports.Wllama = globalThis.pulpTestRuntime.Wllama; exports.CacheManager = globalThis.pulpTestRuntime.CacheManager;");
  const loaderPath = path.join(directory, "llm.cjs");
  copyFileSync(require.resolve("../live/llm.js"), loaderPath);
  const loader = require.resolve(loaderPath);
  t.after(() => { delete require.cache[loader]; delete globalThis.runPulpModel; });
  async function reloadAndSearch() {
    delete require.cache[loader];
    require(loader);
    assert.equal(await globalThis.runPulpModel("grocery query"), "loaded");
  }

  await reloadAndSearch();
  assert.equal(modelRequests, 1, "empty storage needs exactly one model download");
  await reloadAndSearch();
  assert.equal(modelRequests, 1, "a complete cached model must survive a page reload");
  const entry = store.get(await new CacheManager().getNameFromURL(url));
  entry.blob = entry.blob.slice(0, entry.originalSize);
  await reloadAndSearch();
  assert.equal(modelRequests, 2, "a truncated model must be replaced even when its size equals the HTTP header");
});
