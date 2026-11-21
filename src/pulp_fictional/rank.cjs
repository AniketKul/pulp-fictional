// Read the published catalog and its projection once. The server sends one JSON request per line.
const fs = require("node:fs");
const path = require("node:path");
const site = process.argv[2];
require(path.join(site, "retrieval.js"));
const read = name => JSON.parse(fs.readFileSync(path.join(site, "data", name + ".json"), "utf8"));
const source = read("source");
const market = read("market");
const embeddings = read("embeddings");
const historyByQuery = new Map(source.queries.map(row => [row.query, row]));

function words(text) {
  return String(text || "").normalize("NFKD").toLowerCase().replace(/\p{M}/gu, "")
    .match(/[a-z0-9]+|[\p{Script=Han}]+/gu) || [];
}

function mentions(text, phrase) {
  const needle = words(phrase).join(" ");
  const haystack = words(text).join(" ");
  if (!needle) return false;
  if (/\p{Script=Han}/u.test(needle)) return haystack.includes(needle);
  return (" " + haystack + " ").includes(" " + needle + " ");
}

const description = p => [p.title, p.english_title, p.brand, p.category, p.english_category].join(" ");

function rank(req) {
  const history = historyByQuery.get(req.source_query);
  const cached = req.route === "teacher-cache";
  const {product, brand, attributes} = req.proposal;
  // A shopper may use the retail brand printed in the title while the source brand names its parent company.
  const aliases = brand.split(/[/／]/).map(s => s.trim()).filter(Boolean);
  const requested = aliases.filter(alias => mentions(req.query, alias));
  const catalog = requested.length ? {products: market.products.filter(p =>
    requested.some(alias => mentions([p.brand, p.title, p.english_title].join(" "), alias)))} : market;
  const result = globalThis.rankGroceries(catalog, source, embeddings, req.query, req.proposal, history, cached);
  const evidence = [req.query, ...result.products.map(description)];
  const supported = term => evidence.some(text => mentions(text, term));
  const productWords = words(product);
  const roles = cached ? {product, brand, attributes} : {
    product: productWords.length && evidence.some(text => productWords.every(w => mentions(text, w))) ? product : "",
    brand: aliases.some(supported) ? brand : "",
    attributes: attributes.filter(supported),
  };
  const omitted = roles.product !== product || roles.brand !== brand || roles.attributes.length !== attributes.length;
  return {...result, query: req.query, route: req.route, backend: req.backend,
    source_query: history?.query, roles, ...(omitted ? {unverified_roles_omitted: true} : {})};
}

if (process.argv[3] === "--serve") {
  const lines = require("node:readline").createInterface({input: process.stdin, crlfDelay: Infinity});
  lines.on("line", line => {
    try {
      process.stdout.write(JSON.stringify(rank(JSON.parse(line))) + "\n");
    } catch {
      process.stdout.write('{"error":"Catalog ranking failed."}\n');
    }
  });
} else {
  process.stdout.write(JSON.stringify(rank(JSON.parse(fs.readFileSync(0, "utf8")))));
}
