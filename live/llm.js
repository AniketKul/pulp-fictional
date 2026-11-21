/* Classic script, loaded before the storefront. Only a cache miss needs the model. */
(() => {
  const base = window.PULP_MODEL_ASSETS
    ? new URL(window.PULP_MODEL_ASSETS.replace(/\/?$/, "/"), document.baseURI)
    : new URL(".", document.currentScript.src);
  const asset = (name) => new URL(name, base).href;
  let ready;

  async function initialize(onProgress) {
    const [{ Wllama }, response] = await Promise.all([
      import(asset("wllama.js")),
      fetch(asset("model.json"), { cache: "no-cache" }),
    ]);
    if (!response.ok) throw new Error("The language model manifest could not be loaded.");
    const { file, bytes, sha256 } = await response.json();
    if (!/^[0-9a-f]{64}$/.test(sha256) || file !== `student-${sha256.slice(0, 16)}.gguf` ||
        !Number.isSafeInteger(bytes) || bytes <= 0) throw new Error("Invalid language model manifest.");
    const llm = new Wllama({ default: asset("wllama.wasm") }, { suppressNativeLog: true });
    const url = asset(file);
    const cache = llm.cacheManager;
    async function openWeights() {
      const key = await cache.getNameFromURL(url);
      let blob = await cache.open(key);
      // The CDN can gzip the weights. The manifest records the decoded size stored by the browser.
      if (blob?.size === bytes) return blob;
      if (blob) await cache.delete(key);
      await cache.download(url, {
        metadataAdditional: { originalSize: bytes },
        progressCallback: ({ loaded }) => onProgress?.({ loaded, total: bytes }),
      });
      blob = await cache.open(key);
      if (blob?.size !== bytes) {
        await cache.delete(key);
        throw new Error("The downloaded language model is incomplete.");
      }
      return blob;
    }
    const blob = navigator.locks
      ? await navigator.locks.request(url, openWeights)
      : await openWeights();
    await llm.loadModel([blob], {
      n_ctx: 2048,
      n_parallel: 1,
      n_threads: 1,
      n_gpu_layers: 0,
    });
    window.runPulpModel.ready = true;
    return llm;
  }

  window.runPulpModel = async (prompt, { onProgress, signal, maxTokens = 96 } = {}) => {
    if (typeof prompt !== "string" || !prompt.length) throw new Error("A model prompt is required.");
    if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 512) throw new Error("Invalid output token limit.");
    ready ??= initialize(onProgress).catch((cause) => {
      ready = undefined;
      throw new Error("Could not start the browser language model. Check your connection, browser storage, and WebAssembly support, then retry.", { cause });
    });
    const model = await ready;
    signal?.throwIfAborted();
    const completion = await model.createCompletion({
      prompt, max_tokens: maxTokens, temperature: 0, seed: 31, abortSignal: signal,
      stop: ["<|im_end|>", "<|endoftext|>"],
    });
    const text = completion.choices[0]?.text;
    if (!text?.trim()) throw new Error("The language model returned an empty answer.");
    return text;
  };
  window.runPulpModel.ready = false;
})();
