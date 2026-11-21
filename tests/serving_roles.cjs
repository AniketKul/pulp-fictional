const {test} = require("node:test");
const assert = require("node:assert/strict");
const {execFileSync} = require("node:child_process");
const path = require("node:path");

function rank(query, roles) {
  const root = path.join(__dirname, "..");
  const proposal = {product: "", brand: "", attributes: [], synonym: "", substitute: "", broader: "", ...roles};
  return JSON.parse(execFileSync(process.execPath, [
    path.join(root, "src/pulp_fictional/rank.cjs"), path.join(root, "live"),
  ], {input: JSON.stringify({query, proposal, route: "live-student"}), encoding: "utf8"}));
}

test("the explicit retail brand in a listing title constrains the returned products", () => {
  const result = rank("Nescafé instant black coffee", {product: "coffee", brand: "Nescafe"});
  assert.deepEqual(result.products.map(p => p.id), ["869085"]);
});

test("unsupported model attributes are withheld from the shopper's trace", () => {
  const result = rank("a sugar-free mooncake", {product: "mooncake", attributes: ["sugar-free", "automotive"]});
  assert.equal(result.products[0].id, "68170");
  assert.deepEqual(result.roles.attributes, ["sugar-free"]);
  assert.equal(result.unverified_roles_omitted, true);
});
