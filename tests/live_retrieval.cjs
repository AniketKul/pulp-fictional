/* Run after generating live/data/embeddings.json. Only real source rows are used as test data. */
const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const path = require("node:path");
const {createHash} = require("node:crypto");
require("../live/retrieval.js");
const dir = path.resolve(process.argv[2] || path.join(__dirname, "../live/data"));
const load = n => JSON.parse(readFileSync(path.join(dir,n),"utf8"));
const hash = n => createHash("sha256").update(readFileSync(path.join(dir,n))).digest("hex");
const market = load("market.json"), source = load("source.json");
const e = process.argv[3] ? JSON.parse(readFileSync(process.argv[3],"utf8")) : load("embeddings.json");
assert.equal(e.market_sha256, hash("market.json"));
assert.equal(e.source_sha256, hash("source.json"));
assert.equal(e.glossary_sha256, hash("glossary.json"));
assert.equal(e.rank, 64);
const search = (q, proposal = {}, observed = null) =>
  globalThis.rankGroceries(market, source, e, q, proposal, observed, Boolean(observed));
const byId = new Map(source.products.map(p => [p.id,p]));
for (const cached of market.cache) {
  const history = source.queries.find(q => q.query === cached.query);
  assert.ok(history, cached.query);
  const result = search(cached.english, {...cached, ...cached.roles}, history);
  assert.equal(result.evidence.reduce((n,x) => n+x.count, 0), Object.values(history.purchases).reduce((a,b)=>a+b,0));
  assert.ok(result.evidence.every(row => Number.isFinite(row.cosine) && Math.abs(row.cosine) <= 1.001));
  assert.ok(result.products.every(p => byId.has(p.id)));
}
const mooncakes = market.cache.find(r => r.query === "月饼");
assert.ok(mooncakes);
const original = source.queries.find(q => q.query === mooncakes.query);
assert.ok(search(mooncakes.english, mooncakes.roles, original).products.some(p => /月饼/.test(p.title)));
const rewritten = search(mooncakes.english, {...mooncakes, ...mooncakes.roles}, original).rewrites;
assert.ok(![...rewritten.synonyms, ...rewritten.broader].some(q => /sugar[- ]free/i.test(q)));
const malformed = search("Boneless chicken feet", {broader:"Boneless chicken feet hot and sour NONE"}).rewrites;
assert.deepEqual(malformed.broader, []);
const absent = search("wireless charging pad", {product:"moon cake",brand:"Dove",broader:"tea"});
assert.deepEqual(absent.products, []);
assert.deepEqual(absent.evidence, []);
assert.deepEqual(search("apple charger", {product:"apples"}).products, []);
assert.deepEqual(search("dark chocolate gift box for a friend", {product:"chocolate",attributes:["dark"]}).products.map(p => p.id), ["4188"]);
assert.deepEqual(search("dark chocolate for my friend with a peanut allergy", {product:"chocolate"}).products, []);
for (const q of ["不要花生的巧克力", "牛奶过敏巧克力", "不能吃花生的巧克力", "無花生的巧克力"]) {
  assert.deepEqual(search(q, {product:"chocolate"}).products, [], q);
}
for (const query of ["mooncakes without peanuts","gluten-free mooncakes","月饼不含花生"]) {
  const hits = search(query, {product:"moon cake"}).products;
  for (const p of hits) assert.match(p.title + p.category, /不含花生|无花生|无麸质|不含麸质/i);
}
// This listing says "no saccharin" (无糖精). Its seller makes no sugar-free claim.
assert.ok(!search("sugar-free bread", {product:"bread"}).products.some(p => p.id === "769876"));
const corn = search("fresh golden waxy corn", {product:"corn"});
assert.ok(corn.products.some(p => p.id === "482767"), "Fresh golden waxy corn must match the source listing");
assert.deepEqual(corn.evidence, []);
console.log("Verified source-bound embedding vectors, historical counts, real catalog matches, and constraint guardrails.");
