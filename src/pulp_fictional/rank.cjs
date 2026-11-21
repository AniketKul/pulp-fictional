// Local REST searches reuse the web ranker and the same source-bound vectors as the storefront.
const fs = require("node:fs");
const path = require("node:path");
const site = process.argv[2];
require(path.join(site, "retrieval.js"));
const read = name => JSON.parse(fs.readFileSync(path.join(site, "data", name + ".json"), "utf8"));
const req = JSON.parse(fs.readFileSync(0, "utf8"));
const source = read("source");
const history = source.queries.find(row => row.query === req.source_query);
const result = globalThis.rankGroceries(read("market"),source,read("embeddings"),req.query,req.proposal,history,
                                      req.route === "teacher-cache");
const {product,brand,attributes} = req.proposal;
process.stdout.write(JSON.stringify({...result, query:req.query,route:req.route,backend:req.backend,
  source_query:history?.query,roles:{product,brand,attributes}}));
