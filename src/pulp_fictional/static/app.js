"use strict";

const byId = (id) => document.getElementById(id);
const queryInput = byId("query");
const currency = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
let pending;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

// Images represent aisles. New product families use the market-basket illustration.
const aisles = [
  [/^bread-/, "bread"], [/^herb-/, "herb"], [/^oat-/, "oat"],
  [/^dairy-/, "dairy"], [/^smoothie-/, "smoothie"], [/^(freezer|storage)-/, "bag"],
  [/^roast-/, "roast"]
];

function productCard(product) {
  const item = element("li", undefined, "product");
  const aisle = aisles.find(([pattern]) => pattern.test(product.id));
  const illustration = element("span", undefined, "product-img " + (aisle?.[1] || "basket"));
  illustration.setAttribute("aria-hidden", "true");
  const info = element("div", undefined, "product-info");
  info.append(element("p", product.brand, "brand"), element("h3", product.name));
  const meta = element("p", undefined, "product-meta");
  meta.append(element("strong", currency.format(product.price_cents / 100)), element("span", product.size));
  info.append(meta);
  const attributes = element("div", undefined, "attributes");
  for (const value of product.attributes) attributes.append(element("span", value));
  info.append(attributes);
  item.append(illustration, info);
  return item;
}

function displayFields(target, fields) {
  byId(target).replaceChildren(...fields.flatMap(([label, value]) =>
    [element("dt", label), element("dd", value || "—")]));
}

function roleFields(roles = {}) {
  return [["Product", roles.product], ["Brand", roles.brand], ["Attributes", roles.attributes?.join(", ")]];
}

function renderChecks(response) {
  byId("provenance").hidden = !(response.proposed_roles || response.backend || response.retrieval);
  displayFields("model-facts", [["Backend", response.backend], ["Retrieval", response.retrieval]]);
  byId("proposal-label").textContent = response.route === "head-cache"
    ? "Cached roles submitted to online validation"
    : "Student roles submitted to online validation";
  displayFields("proposal", roleFields(response.proposed_roles || {}));
  const candidates = response.candidates || [];
  byId("candidate-details").hidden = !candidates.length;
  byId("candidates").replaceChildren(...candidates.map(candidate => {
    const item = element("li");
    const stats = [];
    if (Number.isFinite(candidate.conversions)) stats.push(candidate.conversions + " conversions");
    if (Number.isFinite(candidate.relevance)) stats.push("relevance " + candidate.relevance.toFixed(2));
    if (Number.isFinite(candidate.cosine)) stats.push("cosine " + candidate.cosine.toFixed(2));
    item.append(element("p", candidate.category.replaceAll(" > ", " › ")));
    if (stats.length) item.append(element("small", stats.join(" · ")));
    if (candidate.accepted === false) item.append(element("small", "Rejected · " + (candidate.guard || "catalog check"), "rejected"));
    else if (candidate.accepted) item.append(element("small", "Accepted", "accepted"));
    return item;
  }));
}

function render(response) {
  const { query, roles, categories, rewrites, products, route } = response;
  byId("products").replaceChildren(...products.map(productCard));
  byId("count").textContent = products.length + (products.length === 1 ? " item" : " items");
  byId("result-query").textContent = 'For “' + query + '”';
  byId("empty").hidden = products.length !== 0;
  const cached = route === "head-cache";
  byId("route").textContent = cached ? "Head query · teacher cache" : "Long-tail query · live student";
  byId("route-detail").textContent = cached
    ? "This query matched a teacher-labeled entry that passed offline catalog validation."
    : "This query missed the cache. The trained student predicted its semantic roles.";
  displayFields("roles", roleFields(roles));
  renderChecks(response);
  byId("categories").replaceChildren(...(categories.length ? categories : ["None matched"]).map(
    (path) => element("li", path.replaceAll(" > ", " › "))));
  const groups = [["Synonyms", rewrites.synonyms], ["Substitutes", rewrites.substitutes], ["Broader", rewrites.broader]];
  byId("rewrites").replaceChildren(...groups.map(([label, values]) => {
    const group = element("div", undefined, "rewrite-group");
    group.append(element("h4", label));
    for (const value of values) {
      const btn = element("button", value);
      btn.type = "button";
      btn.dataset.query = value;
      group.append(btn);
    }
    if (!values.length) group.append(element("span", "None", "muted"));
    return group;
  }));
  byId("analysis").hidden = false;
}

async function search(raw, updateUrl = true, focusResults = false) {
  const query = raw.trim();
  if (!query) {
    queryInput.setCustomValidity("Enter a product, brand, or dietary preference.");
    queryInput.reportValidity();
    return;
  }
  queryInput.setCustomValidity("");
  pending?.abort();
  const request = new AbortController();
  pending = request;
  queryInput.value = query;
  byId("status").textContent = "Searching…";
  const params = new URLSearchParams({ q: query });
  try {
    const result = await fetch("/api/search?" + params, { signal: request.signal });
    if (!result.ok) throw new Error("Search request failed");
    const data = await result.json();
    if (request !== pending) return;
    render(data);
    if (focusResults) byId("market-title").focus();
    if (updateUrl && location.search !== "?" + params) history.pushState(null, "", "?" + params);
    byId("status").replaceChildren(element("span", byId("count").textContent + " found for " + query, "visually-hidden"));
  } catch (error) {
    if (request !== pending || error.name === "AbortError") return;
    byId("status").textContent = "Search is unavailable. Please try again.";
    for (const id of ["products", "count", "route-detail", "result-query"]) byId(id).replaceChildren();
    byId("analysis").hidden = true;
    byId("empty").hidden = true;
    byId("route").textContent = "Trace unavailable";
  }
}

queryInput.addEventListener("input", () => queryInput.setCustomValidity(""));
byId("search").addEventListener("submit", (event) => {
  event.preventDefault();
  search(queryInput.value);
});
document.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-query]");
  if (button) search(button.dataset.query, true, byId("rewrites").contains(button));
});
function loadUrl() {
  search(new URLSearchParams(location.search).get("q")?.trim() || "bread no gluten", false);
}
window.addEventListener("popstate", loadUrl);
loadUrl();
