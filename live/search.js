"use strict";
(() => {
  const normalize = s => s.normalize("NFKC").trim().toLowerCase().replace(/\s+/g, " ");
  let data, queue = Promise.resolve(), active;
  async function getJSON(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error("Could not load " + path + " (" + response.status + ").");
    return response.json();
  }
  function load() {
    data ??= Promise.all(["market", "source", "embeddings", "prompt"].map(name => getJSON("./data/" + name + ".json")))
      .then(([market, source, embeddings, prompt]) => {
        const cached = new Map(), observed = new Map();
        for (const row of source.queries) observed.set(normalize(row.query), row);
        for (const row of market.cache) {
          cached.set(normalize(row.query), row);
          if (row.english) cached.set(normalize(row.english), row);
        }
        return {market, source, embeddings, prompt: prompt.system, cached, observed};
      }).catch(error => { data = undefined; throw error; });
    return data;
  }
  window.pulpExamples = async () => {
    const {market} = await load();
    // Offer different product families from the actual frequency-selected cache.
    const seen = new Set();
    return market.cache.filter(r => {
      const family = r.roles.product.toLowerCase();
      if (seen.has(family)) return false;
      seen.add(family);
      return true;
    }).slice(0, 4).map(r => ({text: r.english || r.query, label: r.english || r.query}));
  };
  window.pulpNeedsModel = async (q, force) => {
    const {cached} = await load();
    return !window.PULP_GPU && !window.runPulpModel.ready && (force || !cached.has(normalize(q)));
  };
  function parse(raw) {
    let obj;
    try { obj = JSON.parse(raw.trim().replace(/^```(?:json)?\s*|\s*```$/g, "")); } catch (_) { /* checked below */ }
    if (!obj || !["product", "brand", "synonym", "substitute", "broader"].every(k => typeof obj[k] === "string")
        || !Array.isArray(obj.attributes) || !obj.attributes.every(x => typeof x === "string")) {
      throw new Error("The student returned an incomplete intent. Try a shorter product query.");
    }
    return obj;
  }
  window.searchGroceries = async (query, progress = () => {}, force = false) => {
    const q = query.trim();
    if (!q || q.length > 200 || /<\|(?:im_(?:end|start)|endoftext)\|>/i.test(q))
      throw new Error("Enter a grocery search of 1 to 200 characters.");
    active?.abort();
    const controller = new AbortController();
    active = controller;
    const {market, source, embeddings, prompt, cached, observed} = await load();
    const head = cached.get(normalize(q));
    const history = observed.get(normalize(head?.query || q));
    let proposed, backend, route;
    if (head && !force) {
      proposed = {...head.roles, synonym: head.synonym, substitute: head.substitute,
                  broader: head.broader, categories: head.categories};
      backend = "Qwen2.5-7B-Instruct · offline";
      route = "teacher-cache";
    } else {
      route = "live-student";
      let raw;
      if (window.PULP_GPU) {
        progress("Running the fine-tuned Qwen model on the GPU…");
        const response = await fetch("./api/intent", {method:"POST", headers:{"Content-Type":"application/json"},
          body:JSON.stringify({query:q}),signal:controller.signal});
        if (!response.ok) throw new Error("GPU inference is unavailable (" + response.status + ").");
        raw = (await response.json()).completion;
        backend = "Qwen3-0.6B · merged LoRA · GPU";
      } else {
        backend = "Qwen3-0.6B · merged LoRA · browser CPU";
        progress("Loading the language model and running your search…");
        const formatted = "<|im_start|>system\n" + prompt + "<|im_end|>\n<|im_start|>user\n" + q +
                          "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        // The browser owns one inference context. Serialize calls and cancel results from superseded queries.
        const job = queue.then(() => {
          controller.signal.throwIfAborted();
          return window.runPulpModel(formatted, {maxTokens:150, signal:controller.signal, onProgress: event => {
            if (event.total) progress("Loading model " + Math.min(100, Math.round(100*event.loaded/event.total)) + "%…");
          }});
        });
        queue = job.catch(() => {});
        raw = await job;
      }
      proposed = parse(raw);
    }
    controller.signal.throwIfAborted();
    const result = window.rankGroceries(market, source, embeddings, q, proposed, history, route === "teacher-cache");
    return {query:q,route,backend,source_query:history?.query,roles:{product:proposed.product,brand:proposed.brand,
      attributes:proposed.attributes},...result};
  };
})();
