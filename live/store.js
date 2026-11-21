"use strict";

(() => {
  const byId = id => document.getElementById(id);
  const input = byId("query");
  const form = byId("search");
  let active;
  let timer;
  let composing = false;
  let currentQuery = "";

  function node(tag, text, className) {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (className) el.className = className;
    return el;
  }

  function facts(id, fields) {
    byId(id).replaceChildren(...fields.filter(([, value]) => value).flatMap(([key, value]) =>
      [node("dt", key), node("dd", Array.isArray(value) ? value.join(", ") : value)]));
  }

  function queryButton(text, label) {
    const button = node("button", label || text);
    button.type = "button";
    button.dataset.query = text;
    return button;
  }

  function productCard(product) {
    const item = node("li");
    if (product.brand) item.append(node("p", product.brand, "brand"));
    item.append(node("h3", product.english_title || product.title, "product-name"));
    if (product.english_title && product.english_title !== product.title) {
      const original = node("p", product.title, "source-title");
      original.lang = "zh";
      item.append(original);
    }
    const meta = node("p", undefined, "product-meta");
    if (product.category) meta.append(node("span", product.english_category || product.category));
    meta.append(node("span", "Source item " + product.id));
    item.append(meta);
    return item;
  }

  function render(result) {
    const cached = result.route === "teacher-cache";
    currentQuery = result.query;
    const products = result.products || [];
    byId("products").replaceChildren(...products.map(productCard));
    byId("empty").hidden = products.length !== 0;
    byId("count").textContent = products.length === 1 ? "1 item" : products.length + " items";
    byId("result-query").textContent = "For “" + result.query + "”";
    byId("route").textContent = cached ? "Teacher cache" : "Live student";
    byId("route-detail").textContent = cached
      ? "This search reused an intent labeled by the offline teacher."
      : result.unverified_roles_omitted
        ? "The student ran on this query. Only tags supported by the query or matching listings are shown."
        : "The student model ran on this query.";
    facts("model", [["Model", result.backend], ["Source query", result.source_query]]);
    byId("rerun").hidden = !cached;
    const roles = result.roles || {};
    facts("roles", [["Product", roles.product || "—"], ["Brand", roles.brand || "—"],
      ["Attributes", roles.attributes?.length ? roles.attributes : "—"]]);
    const categories = result.categories || [];
    const scores = new Map((result.category_scores || []).map(row => [row.category, row.cosine]));
    byId("categories").replaceChildren(...(categories.length ? categories : ["No category matched"]).map(category => {
      const li = node("li", category);
      if (Number.isFinite(scores.get(category))) li.append(node("small", " · cosine " + scores.get(category).toFixed(2)));
      return li;
    }));
    const evidence = result.evidence || [];
    byId("evidence-info").textContent = evidence.length
      ? "Recorded purchases in this sample, grouped by category."
      : "No recorded purchase evidence for this query in the sample.";
    byId("evidence").replaceChildren(...evidence.map(row => {
      const item = node("li", row.category);
      const details = [];
      if (Number.isFinite(row.count)) details.push(row.count + (row.count === 1 ? " purchase" : " purchases"));
      if (Number.isFinite(row.cosine)) details.push("cosine " + row.cosine.toFixed(2));
      if (typeof row.accepted === "boolean") details.push(row.accepted ? "Accepted" : "Rejected");
      if (details.length) item.append(node("small", details.join(" · ")));
      return item;
    }));
    byId("rewrites").replaceChildren(...[["Synonyms", "synonyms"], ["Substitutes", "substitutes"], ["Broader", "broader"]]
      .map(([label, key]) => {
        const section = node("div");
        const options = node("div", undefined, "rewrite-options");
        const values = result.rewrites?.[key] || [];
        options.append(...(values.length ? values.map(v => queryButton(v)) : [node("span", "None", "muted")]));
        section.append(node("h4", label), options);
        return section;
      }));
    byId("analysis").hidden = false;
  }

  function clearResults() {
    currentQuery = "";
    for (const id of ["count", "result-query", "products", "route-detail"]) byId(id).replaceChildren();
    byId("analysis").hidden = true;
    byId("empty").hidden = true;
  }

  function updateUrl(q, replace) {
    const url = new URL(location.href);
    if (q) url.searchParams.set("q", q);
    else url.searchParams.delete("q");
    if (url.href !== location.href) history[replace ? "replaceState" : "pushState"](null, "", url);
  }

  async function search(raw, { force = false, navigation = "push", focus = false } = {}) {
    clearTimeout(timer);
    const q = raw.trim();
    if (!q || q.length > input.maxLength) return;
    input.value = q;
    if (navigation !== "none") updateUrl(q, navigation === "replace");
    if (!force && active?.query === q) return;
    if (!force && !active && currentQuery === q) {
      byId("status").textContent = "";
      return;
    }
    active?.controller.abort();
    const controller = new AbortController();
    active = { query: q, controller };
    const url = new URL(form.action);
    url.searchParams.set("q", q);
    if (force) url.searchParams.set("force", "1");
    byId("status").textContent = "Searching…";
    byId("market").setAttribute("aria-busy", "true");
    try {
      const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(25000)]);
      const response = await fetch(url, { signal, credentials: "omit", cache: "no-store" });
      if (!response.ok) {
        if (response.status === 400) throw new Error("Try a product, brand or ingredient in 1 to 200 characters.");
        if (response.status === 429) throw new Error("The market is busy. Try again in a moment.");
        throw new Error("The search service is unavailable. Please try again.");
      }
      const result = await response.json();
      if (controller.signal.aborted) return;
      if (result.query !== q || !["teacher-cache", "live-student"].includes(result.route) ||
          !Array.isArray(result.products)) throw new Error("The search service returned an incomplete result.");
      render(result);
      byId("status").replaceChildren(node("span", byId("count").textContent + " found for " + q, "sr"));
      if (focus) byId("market-title").focus();
    } catch (error) {
      if (controller.signal.aborted) return;
      clearResults();
      byId("route").textContent = "Search unavailable";
      byId("status").textContent = error.name === "TimeoutError" ? "Search timed out. Please try again."
        : error instanceof TypeError ? "Cannot reach the search service. Please try again." : error.message;
    } finally {
      if (active?.controller === controller) {
        active = undefined;
        byId("market").setAttribute("aria-busy", "false");
      }
    }
  }

  function loadUrl() {
    const q = new URLSearchParams(location.search).get("q")?.trim();
    search(q || byId("examples").querySelector("button").dataset.query, { navigation: "none" });
  }

  function queueSearch() {
    clearTimeout(timer);
    active?.controller.abort();
    active = undefined;
    byId("market").setAttribute("aria-busy", "false");
    const q = input.value.trim();
    if (!q) {
      clearResults();
      byId("route").textContent = "Waiting for search";
      byId("status").textContent = "";
      updateUrl("", true);
      return;
    }
    if (!composing) timer = setTimeout(() => search(q, { navigation: "replace" }), 350);
  }

  form.addEventListener("submit", event => {
    event.preventDefault();
    if (!composing) search(input.value);
  });
  input.addEventListener("input", queueSearch);
  input.addEventListener("compositionstart", () => { composing = true; clearTimeout(timer); });
  input.addEventListener("compositionend", () => { composing = false; queueSearch(); });
  byId("rerun").addEventListener("click", () => search(currentQuery, { force: true }));
  document.addEventListener("click", event => {
    const button = event.target.closest("button[data-query]");
    if (button) search(button.dataset.query, { focus: byId("rewrites").contains(button) });
  });
  window.addEventListener("popstate", loadUrl);
  loadUrl();
})();
