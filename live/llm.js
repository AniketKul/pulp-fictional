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
    const { file } = await response.json();
    const llm = new Wllama({ default: asset("wllama.wasm") }, { suppressNativeLog: true });
    await llm.loadModelFromUrl(asset(file), {
      n_ctx: 2048,
      n_parallel: 1,
      n_threads: 1,
      n_gpu_layers: 0,
      progressCallback: onProgress,
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
