"""
Run an LLM base-aisle labeler. Two modes:

  # bake-off: label the validation items and score vs real user intent
  python3 run_labeler.py --model Qwen/Qwen3.6-27B --score label_val.tsv

  # full run: label the whole universe -> <out>
  python3 run_labeler.py --model google/gemma-4-12B-it --input universe.txt --out base_labels.tsv

Big models (>20B params-ish) load in 8-bit to fit the A6000; smaller in bf16.
"""
import sys, re, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE_AISLES = ["produce","dairy","meat","seafood","bakery","baking","spices",
               "grocery","condiments","beverages","liquor","nonfood"]
# fuzzy pantry cluster -> merged for the "honest" secondary metric
PANTRY = {"grocery","spices","baking","condiments","bakery"}

RUBRIC = """You sort grocery shopping-list items into the ONE aisle where a shopper finds that item's core product. Classify by the item's SEMANTIC FOOD TYPE and IGNORE whether it is fresh/frozen/canned/dried/jarred (preservation state is handled separately).

AISLES:
- produce: fresh fruits, vegetables, salad, mushrooms, fresh herbs, fresh garlic/ginger/onion, lemons/limes. (frozen peas, canned corn, canned tomatoes -> produce; state is ignored)
- dairy: milk, cheese, butter, yogurt, cream, sour cream, eggs, margarine.
- meat: beef, pork, poultry/chicken, lamb, sausage, bacon, ham, deli/lunch meat, ground meat, hot dogs.
- seafood: fish, salmon, tuna, shrimp, crab, lobster, scallops, clams, squid, anchovies (fresh OR canned -> seafood).
- bakery: READY-TO-EAT baked goods: bread, rolls, buns, bagels, croissants, tortillas, cakes, pies, pastries, muffins, donuts.
- baking: raw BAKING INGREDIENTS you bake WITH: flour, sugar (all kinds), baking soda/powder, yeast, vanilla/almond extract, cocoa, chocolate chips, cornstarch, food coloring, cake mix, gelatin.
- spices: dried/ground herbs & spices & seasonings: salt, pepper, paprika, cumin, cinnamon, oregano, garlic powder, chili powder, bouillon cubes.
- grocery: shelf-stable pantry staples not covered by another aisle: pasta, rice, grains, oats, cereal, dry/canned beans, lentils, nuts, seeds, dried fruit, crackers, chips, snacks, popcorn, canned soup, broth, tomato paste, tofu.
- condiments: oils, cooking spray, vinegars, sauces (soy, hot, BBQ, pasta sauce, salsa), dressings, ketchup, mustard, mayo, syrups, honey, jam, peanut butter, nut butters, spreads.
- beverages: non-alcoholic drinks: water, juice, soda, coffee, tea, sports/energy drinks, drink mixes.
- liquor: alcoholic drinks: wine, beer, spirits, liqueurs, cider, cooking wine/sherry.
- nonfood: anything not eaten/drunk: cleaning, paper goods, foil/wrap, toiletries, cosmetics, vitamins/supplements/medicine, pet food, baby care, batteries, candles.

RULES:
- Pick exactly ONE aisle from that list. Lowercase.
- Ignore quantities, brands, prep notes. Items may be in ANY language -- understand then classify.
- Between grocery and condiments/spices/baking, prefer the more specific one that fits the product.
- Non-grocery personal notes or gibberish -> nonfood."""

ITEMS_PER_PROMPT = 20
PROMPTS_PER_BATCH = 12


def build_prompt(items):
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(items))
    return (RUBRIC + "\n\nClassify each item below. Output ONE line per item, "
            "EXACTLY `<number>\\t<aisle>` (number, a TAB, the aisle word), nothing else.\n\n"
            + numbered)


def parse(text, n):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s*[\t\|:.\)]+\s*([a-zA-Z]+)", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        a = m.group(2).strip().lower()
        if 0 <= idx < n and a in BASE_AISLES:
            out[idx] = a
    return out


def load(model_name, bits):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    kw = dict(dtype=torch.bfloat16)
    if bits == 8:
        kw = dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", **kw).eval()
    return tok, model


def label_all(items, tok, model, out_path=None):
    groups = [items[i:i+ITEMS_PER_PROMPT] for i in range(0, len(items), ITEMS_PER_PROMPT)]
    results = {}
    fout = open(out_path, "w", encoding="utf-8") if out_path else None
    done = 0
    for b in range(0, len(groups), PROMPTS_PER_BATCH):
        bg = groups[b:b+PROMPTS_PER_BATCH]
        texts = [tok.apply_chat_template([{"role": "user", "content": build_prompt(g)}],
                 tokenize=False, add_generation_prompt=True) for g in bg]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=ITEMS_PER_PROMPT*8,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        dec = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for gi, (grp, txt) in enumerate(zip(bg, dec)):
            base = (b+gi)*ITEMS_PER_PROMPT
            parsed = parse(txt, len(grp))
            for j, item in enumerate(grp):
                a = parsed.get(j, "unknown")
                results[base+j] = a
                if fout:
                    fout.write(f"{item}\t{a}\n")
        done += sum(len(g) for g in bg)
        if fout:
            fout.flush()
        print(f"  {done}/{len(items)} ({done/len(items)*100:.1f}%)", flush=True)
    if fout:
        fout.close()
    return [results[i] for i in range(len(items))]


def score(preds, items, truth):
    from collections import Counter, defaultdict
    n = len(items)
    exact = merged = unknown = 0
    by = defaultdict(lambda: [0, 0])
    conf = Counter()
    pm = lambda a: "PANTRY" if a in PANTRY else a
    for p, t in zip(preds, truth):
        if p == "unknown":
            unknown += 1
        exact += (p == t)
        merged += (pm(p) == pm(t))
        by[t][0] += (p == t); by[t][1] += 1
        if p != t:
            conf[(t, p)] += 1
    print(f"\n== {n} items ==  exact={exact/n*100:.1f}%  pantry-merged={merged/n*100:.1f}%  unknown={unknown}")
    print("per user-aisle (exact):")
    for a in sorted(by):
        c, tt = by[a]
        print(f"  {a:11s} {c:4d}/{tt:4d}  {c/tt*100:5.1f}%")
    print("top confusions (user->llm):")
    for (t, p), k in conf.most_common(12):
        print(f"  {t:11s} -> {p:11s} {k}")
    return exact/n, merged/n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=0, help="8 or 16; 0=auto")
    ap.add_argument("--input")
    ap.add_argument("--out")
    ap.add_argument("--score", help="tsv of item<TAB>truth to score against")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.score:
        pairs = [l.rstrip("\n").split("\t") for l in open(a.score, encoding="utf-8") if "\t" in l]
        if a.limit:
            pairs = pairs[:a.limit]
        items = [p[0] for p in pairs]; truth = [p[1] for p in pairs]
    else:
        items = [l.strip() for l in open(a.input, encoding="utf-8") if l.strip()]
        if a.limit:
            items = items[:a.limit]
        truth = None

    bits = a.bits or (8 if re.search(r"27B|32B|31B|30B|35B", a.model) else 16)
    print(f"loading {a.model} in {bits}-bit ...", flush=True)
    tok, model = load(a.model, bits)
    preds = label_all(items, tok, model, a.out)
    if truth:
        score(preds, items, truth)


if __name__ == "__main__":
    main()
