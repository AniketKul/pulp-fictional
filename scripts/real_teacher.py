"""Ground an offline Qwen teacher in observed grocery searches and purchased items."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-7B-Instruct"
REV = "a09a35458c702b33eeacc393d103063234e8bc28"
RELATIONS = {
    "synonym": "Give one alternative search with the same meaning and constraints.",
    "substitute": "Give one different grocery product that could serve the same use. Preserve any dietary restrictions.",
    "broader": "Give one broader grocery search that includes the request. Preserve explicit dietary restrictions.",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "live/data/source.json")
    parser.add_argument("--glossary", type=Path, default=ROOT / "live/data/glossary.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.source.read_text())
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16).to("cuda").eval()
    def run(task, rows, length=80, demos=()):
        answers = []
        for start in range(0, len(rows), 16):
            chats = []
            for row in rows[start:start + 16]:
                chat = [{"role":"system","content":"Follow the user's extraction or translation instructions exactly."}]
                for source,answer in demos:
                    chat += [{"role":"user","content":task + "\nInput: " + source},
                             {"role":"assistant","content":answer}]
                value = row if isinstance(row,str) else json.dumps(row, ensure_ascii=False)
                chat.append({"role":"user","content":task+"\nInput: "+value+"\nAnswer:"})
                chats.append(tok.apply_chat_template(chat,tokenize=False,add_generation_prompt=True))
            inputs = tok(chats, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=length, do_sample=False,
                                        pad_token_id=tok.pad_token_id)
            answers.extend(s.strip() for s in tok.batch_decode(output[:, inputs.input_ids.shape[1]:],
                                                         skip_special_tokens=True))
        return answers

    def clean_text(text, limit=180):
        text = text.strip().strip('"').strip()
        return text if len(text) <= limit and not any(c in text for c in "\n{}<>") else ""

    def english_ok(text):
        return bool(re.search("[A-Za-z]", text)) and not re.search(r"[\u3400-\u9fff]",text)

    # These English glosses were translated and checked against original fields independently of purchases.
    gloss = json.loads(args.glossary.read_text())
    products = data["products"]
    levels = gloss["levels"]
    catalog = [{**p, "english_title":gloss["titles"][p["id"]],
                "english_category":" > ".join(levels[c.strip()] for c in p["category"].split(">"))}
               for p in products]
    queries = data["queries"]
    english = [gloss["queries"][q["query"]] for q in queries]
    if not all(english_ok(s) for s in [*english, *levels.values(), *gloss["titles"].values()]):
        raise ValueError("Incomplete English glossary")
    by_id = {p["id"]:p for p in catalog}
    contexts = []
    for q,en in zip(queries,english):
        skus = sorted(q["purchases"],key=lambda sku:(-q["purchases"][sku],sku))[:8]
        def words(value):
            return {v for v in re.findall("[a-z0-9]+",value.lower()) if len(v)>1}
        wanted = words(en)
        similar = sorted(catalog, key=lambda p:(-len(wanted & words(p["english_title"])),p["id"]))
        for p in similar[:3]:
            if p["id"] not in skus and wanted & words(p["english_title"]):
                skus.append(p["id"])
        contexts.append({"query":en,"evidence":[
            {"title":by_id[s]["english_title"],"category":by_id[s]["english_category"],
             "purchasing_sessions":q["purchases"].get(s)} for s in skus]})
    tags = run('Return JSON {"product":"","brand":"","attributes":[]} describing only the query IN ENGLISH. '
               'Product is a short noun phrase. Include only brand and attributes requested by the shopper. '
               'Purchases are context, never copy a brand, pack size or flavor from a purchase. '
               'If the query is fruit, the product is fruit, even if it led to a melon. Never infer a narrower '
               'product than the query supports. Opaque queries may have empty roles.', contexts)
    reranks = run('Rank the supplied relevant category paths for this query; exclude irrelevant purchases. '
                  'Return only a JSON array of exact English category strings copied from the evidence.',contexts,110)
    rewrites = {k:run(task+" Return just the short English search phrase, or NONE if no sound proposal exists.",
                      contexts,36) for k,task in RELATIONS.items()}
    def parse(text,default):
        try:
            return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$","",text).strip())
        except ValueError:
            return default
    rows=[]
    for i,(query,context) in enumerate(zip(queries,contexts)):
        proposed = parse(tags[i],{})
        if not isinstance(proposed,dict):
            proposed={}
        product = proposed.get("product")
        attrs = proposed.get("attributes")
        brand = proposed.get("brand")
        valid = (isinstance(product,str) and bool(clean_text(product,80)) and isinstance(brand,str)
                 and english_ok(product) and bool(english[i])
                 and isinstance(attrs,list) and all(isinstance(s,str) for s in attrs))
        if valid:
            words = set(re.findall(r"[a-z0-9]+", product.casefold())) - {"of","and","the","for"}
            grounding = {w.rstrip("s") for w in re.findall(r"[a-z0-9]+",json.dumps(context).lower())}
            valid = any(w.rstrip("s") in grounding for w in words)
        roles = {"product":clean_text(product,80) if valid else "", "brand":clean_text(brand,70) if valid else "",
                 "attributes":[clean_text(a,50) for a in (attrs or []) if valid and clean_text(a,50) and a.casefold() in english[i].casefold()] if valid else []}
        # A teacher can copy a brand from a purchase: require query text evidence in either language.
        if roles["brand"] and roles["brand"].casefold() not in (query["query"]+" "+english[i]).casefold():
            roles["brand"]=""
        options={p["category"] for p in context["evidence"]}
        ranked=parse(reranks[i],[])
        ranked=list(dict.fromkeys(s for s in ranked if isinstance(s,str) and s in options)) if isinstance(ranked,list) else []
        rel={k:clean_text(v[i],90) for k,v in rewrites.items()}
        rel={k:(v if english_ok(v) and v.upper() not in {"NONE","N/A"} and v.casefold()!=english[i].casefold() else "") for k,v in rel.items()}
        rows.append({"query":query["query"],"english":english[i],"sessions":query["sessions"],
                     "roles":roles,"categories":ranked,**rel,"valid":valid,
                     "raw":{"roles":tags[i],"categories":reranks[i],**{k:v[i] for k,v in rewrites.items()}}})
    ordered=sorted((r for r in rows if r["valid"] and r["english"] and r["categories"]),
                   key=lambda r:(-r["sessions"],r["query"]))
    # Keep head translations unambiguous; exact original phrases remain the source of the evidence.
    english_counts=Counter(r["english"].casefold() for r in rows)
    heads=[r for r in ordered if english_counts[r["english"].casefold()]==1][:12]
    if len(heads)<3 or sum(r["valid"] for r in rows)<20:
        raise ValueError("Insufficient grounded teacher output")
    save(args.out/"market.json", {"products":catalog, "cache":[{k:v for k,v in r.items() if k not in ("raw","valid")}
                                                           for r in heads]})
    save(args.out/"teacher.json", {"model":MODEL,"revision":REV,"source_sha256":digest(args.source),
         "glossary_sha256":digest(args.glossary),"builder_sha256":digest(Path(__file__)),"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),
         "rows":rows})
    print(f"Teacher completed {len(rows)} queries, {sum(r['valid'] for r in rows)} structured labels, {len(heads)} head keys.",flush=True)


if __name__=="__main__":
    main()
