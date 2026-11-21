// Validate all offline rewrite labels using the same real catalog, restrictions and ranker as live search.
const fs = require("node:fs");
require("../live/retrieval.js");
const {products, source, index, rows} = JSON.parse(fs.readFileSync(0, "utf8"));
const observed = new Map(source.queries.map(row => [row.query, row]));
const results = rows.map(row => {
  const proposal = {...row.roles, categories: row.categories, synonym: row.synonym,
                    substitute: row.substitute, broader: row.broader};
  const {rewrites} = globalThis.rankGroceries({products}, source, index, row.english, proposal,
                                             observed.get(row.query), true);
  return {synonym: rewrites.synonyms[0] || "", substitute: rewrites.substitutes[0] || "",
          broader: rewrites.broader[0] || ""};
});
process.stdout.write(JSON.stringify(results));
