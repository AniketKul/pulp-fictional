/* Live catalog search. The HTTP and in-browser model paths use the same ranker. */
(() => {
  const stop = new Set("a an and are at by can could do for from get how i in is it me my of on or please some that the to want with would you looking look find show buy need like one have something shop shopping".split(" "));
  const normalize = value => String(value || "").normalize("NFKC").toLowerCase();
  function tokens(text) {
    const result = [];
    for (let word of normalize(text).match(/[a-z0-9]+|[\u3400-\u9fff]+/g) || []) {
      if (/[\u3400-\u9fff]/.test(word)) {
        result.push(...word);
        for (let i = 0; i < word.length - 1; ++i) result.push(word.slice(i, i + 2));
      } else {
        if (word.length > 3 && word.endsWith("s") && !/(ss|us)$/.test(word)) word = word.slice(0, -1);
        if (!stop.has(word)) result.push(word);
      }
    }
    return result;
  }

  const vec = (dimension) => new Float64Array(dimension);
  const dot = (x, y) => x.reduce((v, a, i) => v + a * y[i], 0);
  const indices = new WeakMap();
  function encode(text, e) {
    if (!indices.has(e)) indices.set(e, new Map(e.vocabulary.map((w, i) => [w, i])));
    const counts = new Map();
    for (const w of tokens(text)) counts.set(w, (counts.get(w) || 0) + 1);
    const out = vec(e.rank);
    for (const [w, n] of counts) {
      const pos = indices.get(e).get(w);
      if (pos === undefined) continue;
      const weight = Math.log1p(n) * e.idf[pos];
      const axis = e.basis[pos];
      for (let i = 0; i < out.length; ++i) out[i] += weight * axis[i];
    }
    const length = Math.sqrt(dot(out, out)) || 1;
    return out.map(v => v / length);
  }

  // A dietary claim needs an explicit source label. Missing allergen information is unknown.
  const restrictions = [
    [/\b(?:no added sugar|without added sugar|no sugar added|unsweetened)\b|无添加糖/i, /无添加糖|不添加糖|不加糖|无糖(?!精)|零糖|\b(?:no added sugar|sugar[- ]free|unsweetened)\b/i],
    [/\b(?:sugar[- ]free|(?:no|without|zero) sugar)\b|无糖(?!精)|零糖|不含糖(?!精)/i, /无糖(?!精)|零糖|不含糖(?!精)|\b(?:sugar[- ]free|zero sugar)\b/i],
    [/\blow[- ]sugar\b|低糖/i, /低糖|无糖(?!精)|零糖|\blow[- ]sugar\b/i],
    [/\blow[- ]sodium\b|低钠/i, /低钠|\blow[- ]sodium\b/i],
    [/\b(?:fat[- ]free|(?:no|without|zero) fat)\b|脱脂|零脂/i, /无脂|脱脂|零脂|\bfat[- ]free\b/i],
    [/\b(?:gluten[- ]free|(?:no|without) gluten)\b|无麸质/i, /无麸质|不含麸质|\bgluten[- ]free\b/i],
    [/\b(?:dairy[- ]free|(?:no|without) dairy)\b|无乳制品/i, /无乳制品|不含乳制品|\bdairy[- ]free\b/i],
    [/\b(?:lactose[- ]free|(?:no|without) lactose)\b|无乳糖/i, /无乳糖|零乳糖|不含乳糖|\blactose[- ]free\b/i],
    [/\b(?:nut[- ]free|(?:no|without) nuts)\b|不含坚果/i, /无坚果|不含坚果|\bnut[- ]free\b/i],
    [/\b(?:peanut[- ]free|(?:no|without) peanuts?)\b|不含花生/i, /不含花生|无花生|\bpeanut[- ]free\b/i],
    [/\b(?:caffeine[- ]free|(?:no|without) caffeine)\b|无咖啡因/i, /无咖啡因|不含咖啡因|\bcaffeine[- ]free\b/i],
    [/\bvegan\b|纯素/i, /纯素|全素|\bvegan\b/i],
    [/\bvegetarian\b|素食/i, /素食|纯素|全素|\bvegetarian\b/i],
    [/\b(?:unsalted|salt[- ]free|(?:no|without) salt)\b|无盐/i, /无盐|不加盐|\b(?:unsalted|salt[- ]free)\b/i],
    [/\bboneless\b|无骨|脱骨|去骨/i, /无骨|脱骨|去骨|\bboneless\b/i],
    [/\borganic\b|有机/i, /有机|\borganic\b/i],
    [/\bhalal\b|清真/i, /清真|\bhalal\b/i],
  ];

  function requirements(query) {
    let rest = normalize(query);
    const required = [];
    for (const [pattern, label] of restrictions) {
      if (!pattern.test(rest)) continue;
      required.push(label);
      rest = rest.replace(pattern, " ");
    }
    // Match unresolved exclusions in each storefront language before applying semantic retrieval.
    // Unknown allergen status must not admit a partially matching product containing that allergen.
    const unsupported = /\b(?:except|without|exclude|excluding|free of|not|no|allerg\w*|\w+[- ]free)\b|(?:^|\s)[−–-]\w/.test(rest)
      || /[无無]|不(?:要|能|可|吃|喝|加|含|带|帶|放|用|想|喜欢|喜歡)|没(?:有|加)|沒(?:有|加)|避(?:免|开|開)|[过過]敏|忌口|禁忌|去(?:掉|除)|排除|剔除|不耐受|零(?:糖|脂|蔗糖|乳糖)|0(?:糖|脂|蔗糖|乳糖)/.test(rest);
    // A gift recipient describes the occasion, while modifiers beside it can still constrain an item.
    rest = rest.replace(/\bfor (?:a |an |the |my |our |your |his |her |their )?(?:friends?|partners?|colleagues?|coworkers?|parents?|siblings?|mothers?|fathers?|sisters?|brothers?|wives|wife|husbands?)\b/g, " ");
    return {required, unsupported, rest};
  }

  const extras = new Set("recommend recommended branded morning afternoon evening late night tonight today enjoy eating breakfast lunch dinner craving tasty delicious maybe great good best quick everyday pack packet bottle bag box style would give anything me grocery product".toLowerCase().split(" "));
  function anchors(text) {
    const all = tokens(text);
    const singleHan = all.length === 1;
    return [...new Set(all.filter(w => !extras.has(w) && (singleHan || !/^[\u3400-\u9fff]$/.test(w))))];
  }

  function entries(products) {
    return products.map(p => {
      const leaf = value => String(value || "").split(/\s*>\s*/).pop();
      const searchable = [p.title, p.english_title, p.brand, leaf(p.category), leaf(p.english_category)].join(" ");
      return {p, terms: new Set(tokens(searchable)), raw: p.title + " " + p.category,
              brands: normalize(p.brand).split(/[/／]/).filter(v => v && v.length > 1 && v !== "unknown")};
    });
  }

  function rankGroceries(market, source, e, query, proposal, observed, cached) {
    const catalog = entries(market.products);
    observed = observed && source.queries.find(row => row.query === observed.query);
    const required = requirements(query);
    const searches = [required.rest];
    if (observed?.query && normalize(observed.query) !== normalize(query)) searches.push(observed.query);
    const counts = new Map();
    const catalogById = new Map(market.products.map(p => [p.id, p]));
    for (const [sku, n] of Object.entries(observed?.purchases || {})) {
      const p = catalogById.get(sku);
      if (p) counts.set(p.category, (counts.get(p.category) || 0) + n);
    }
    const known = String(query || "").toLowerCase().normalize("NFKC");
    const brands = new Set(catalog.flatMap(row => row.brands).filter(b =>
      /[\u3400-\u9fff]/.test(b) ? known.includes(b) : tokens(known).includes(b)));
    const categoryScores = new Map();
    const projected = encode(searches.join(" "), e);
    for (const [c, v] of Object.entries(e.categories)) categoryScores.set(c, dot(projected, v));
    const suggested = Array.isArray(proposal?.categories) ? proposal.categories : [];
    const proposed = encode([proposal?.product, proposal?.brand, ...(Array.isArray(proposal?.attributes) ? proposal.attributes : [])].join(" "), e);
    function hits(texts, limits, exactBrands) {
      if (limits.unsupported) return [];
      const groups = texts.map(anchors).filter(words => words.length);
      if (!groups.length) return [];
      const vector = encode(texts.join(" "), e);
      const wordVectors = new Map([...new Set(groups.flat())].map(w => [w, encode(w, e)]));
      return catalog.flatMap(({p, terms, raw, brands: productBrands}) => {
        if (limits.required.some(label => !label.test(raw))) return [];
        if ([...exactBrands].some(brand => !productBrands.includes(brand))) return [];
        const coverage = Math.max(...groups.map(words => {
          // Each meaningful English word must have catalog support; a single fruit name cannot ground "apple charger".
          if (words.every(w => /^[a-z0-9]+$/.test(w)) &&
              words.some(w => !terms.has(w) && dot(wordVectors.get(w), e.items[p.id]) < 0.28)) return 0;
          return words.filter(w => terms.has(w)).length / words.length;
        }));
        const item = dot(vector, e.items[p.id]);
        const path = dot(vector, e.categories[p.category]);
        if (coverage < 0.34 || item < 0.10 || path < 0.08) return [];
        return [{p, score: coverage * 0.45 + item * 0.4 + path * 0.15}];
      });
    }
    const found = hits(searches, required, brands);
    for (const row of found) {
      const hint = Math.max(0, dot(proposed, e.items[row.p.id]));
      const purchased = observed?.purchases?.[row.p.id] || 0;
      const rankedAt = suggested.findIndex(path => path === row.p.category || path === row.p.english_category);
      row.score += hint * 0.15 + Math.min(0.06, Math.log1p(purchased) * 0.015)
                   + (rankedAt >= 0 ? 0.04 / (1 + rankedAt) : 0);
    }
    found.sort((a, b) => b.score - a.score || String(a.p.id).localeCompare(String(b.p.id)));
    const products = found.map(row => row.p);
    const paths = [...new Set(products.map(p => p.category))];
    const pathLabel = path => catalog.find(({p}) => p.category === path)?.p.english_category || path;
    const evidence = [...counts].map(([path, count]) => ({category: pathLabel(path), count, accepted: paths.includes(path),
      cosine: +categoryScores.get(path).toFixed(3)})).sort((a, b) => b.count - a.count);
    const category_scores = paths.map(path => ({category: pathLabel(path), cosine: +categoryScores.get(path).toFixed(3)}));
    const rewrites = {synonyms: [], substitutes: [], broader: []};
    if (!products.length) return {products, categories: [], category_scores, evidence, rewrites};
    const originals = new Set(products.map(p => p.id));
    const seen = new Set([normalize(query)]);
    for (const [kind, output] of [["synonym", "synonyms"], ["substitute", "substitutes"], ["broader", "broader"]]) {
      const suggestions = Array.isArray(proposal?.[kind]) ? proposal[kind] : [proposal?.[kind]];
      for (const text of suggestions) {
        if (typeof text !== "string" || !text.trim() || text.length > 200 || seen.has(normalize(text))) continue;
        if (/\b(?:none|null|undefined|unknown|n\/a)\b/i.test(text)) continue;
        const candidate = requirements(text);
        if (candidate.unsupported || required.required.some(label => !candidate.required.includes(label))) continue;
        if (kind !== "substitute" && candidate.required.some(label => !required.required.includes(label))) continue;
        if ([...brands].some(brand => !normalize(text).includes(brand))) continue;
        const matches = hits([candidate.rest], candidate, brands).sort((a, b) => b.score - a.score);
        if (!matches.length) continue;
        const ids = new Set(matches.map(row => row.p.id));
        const overlap = [...ids].filter(id => originals.has(id)).length;
        if (kind === "synonym" && (overlap !== ids.size || overlap !== originals.size)) continue;
        if (kind === "broader" && !(overlap === originals.size && ids.size > originals.size)) continue;
        if (kind === "substitute") {
          const part = p => p.category.split(" > ").slice(0, 2).join(" > ");
          if (originals.has(matches[0].p.id) || !products.some(p => part(p) === part(matches[0].p))) continue;
        }
        seen.add(normalize(text)); rewrites[output].push(text);
      }
    }
    return {products, categories: [...new Set(paths.map(pathLabel))], category_scores, evidence, rewrites};
  }
  globalThis.rankGroceries = rankGroceries;
})();
