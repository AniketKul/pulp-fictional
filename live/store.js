"use strict";

(() => {
  const byId = id => document.getElementById(id);
  const input = byId("query");
  let generation = 0;
  let confirmation;
  let currentQuery = "";
  let firstExample = "";

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

  async function search(raw, { force = false, approved = false, updateUrl = true, focus = false } = {}) {
    const q = raw.trim();
    if (!q || q.length > input.maxLength) {
      input.setCustomValidity("Enter a search between 1 and 200 characters.");
      input.reportValidity();
      return;
    }
    input.setCustomValidity("");
    const token = ++generation;
    confirmation = undefined;
    byId("notice").hidden = true;
    input.value = q;
    byId("status").textContent = "Checking the query…";
    try {
      const needsModel = !approved && await window.pulpNeedsModel(q, force);
      if (token !== generation) return;
      if (needsModel) {
        confirmation = () => search(q, { force, approved: true, updateUrl, focus });
        byId("status").textContent = "Waiting for permission to load the browser model.";
        byId("notice").hidden = false;
        byId("approve").focus();
        return;
      }
      clearResults();
      byId("route").textContent = "Waiting for search";
      byId("status").textContent = "Searching…";
      const result = await window.searchGroceries(q, progress => {
        if (token === generation && typeof progress === "string") byId("status").textContent = progress;
      }, force);
      if (token !== generation) return;
      render(result);
      const params = new URLSearchParams({ q });
      if (updateUrl && location.search !== "?" + params) history.pushState(null, "", "?" + params);
      byId("status").replaceChildren(node("span", byId("count").textContent + " found for " + q, "sr"));
      if (focus) byId("market-title").focus();
    } catch (error) {
      if (token !== generation) return;
      clearResults();
      byId("route").textContent = "Search unavailable";
      byId("status").textContent = error?.message
        ? "Search failed. " + error.message
        : "Search failed. Please try again.";
    }
  }

  function loadUrl() {
    const q = new URLSearchParams(location.search).get("q")?.trim() || firstExample;
    if (q) search(q, { updateUrl: false });
  }

  byId("search").addEventListener("submit", event => {
    event.preventDefault();
    search(input.value);
  });
  input.addEventListener("input", () => {
    input.setCustomValidity("");
    if (confirmation) {
      confirmation = undefined;
      byId("notice").hidden = true;
      byId("status").textContent = "";
    }
  });
  byId("approve").addEventListener("click", () => confirmation?.());
  byId("cancel").addEventListener("click", () => {
    confirmation = undefined;
    byId("notice").hidden = true;
    byId("status").textContent = "Model load canceled. Choose an example to try a cached query.";
    input.focus();
  });
  byId("rerun").addEventListener("click", () => search(currentQuery, { force: true }));
  document.addEventListener("click", event => {
    const button = event.target.closest("button[data-query]");
    if (button) search(button.dataset.query, { focus: byId("rewrites").contains(button) });
  });
  window.addEventListener("popstate", loadUrl);

  async function start() {
    byId("status").textContent = "Loading the grocery catalog…";
    try {
      const examples = await window.pulpExamples();
      const choices = examples.filter(e => typeof e.text === "string" && e.text.trim());
      firstExample = choices[0]?.text || "";
      byId("examples").append(...choices.map(e => queryButton(e.text, e.label)));
      byId("examples").hidden = !choices.length;
      if (!generation && !choices.length) byId("status").textContent = "Enter a query to search.";
    } catch (error) {
      if (!generation) byId("status").textContent = "Could not load example searches. Try entering a query.";
    }
    if (!generation && !input.value) loadUrl();
  }
  start();
})();
