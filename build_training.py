"""
Assemble the factorized training set.

Base labels (Gemma) are keyed by the cue-STRIPPED form (light_normalize), and
are state-agnostic. The MODEL INPUT, however, must retain the state cue
(can/jar/frozen/canned) so the state head can learn it -- so we reprocess the
raw sources with normalize_model (cue-KEEPING) for the text, and look up the
base label via light_normalize (cue-STRIPPED) key. No re-labeling needed.

Sources:
  data.txt                -> LLM base labels (weight 1)
  real_trainfold.tsv      -> real user overrides, ALL aisles (weight 4); base aisles
                             supervise the base head (state=none), frozen/canned
                             supervise the state head (base from the labeler).
Excludes held-out test items and a real-override val fold (for checkpoint selection).
Outputs:
  train.tsv     model_text<TAB>base<TAB>state<TAB>weight
  real_val.tsv  final_aisle____title   (held-out override fold, ~14%)
"""
import collections, random, hashlib, re
from state_utils import detect_state
from aisle_map import BASE_AISLES, FINAL_AISLES
from build_universe import light_normalize, normalize_model

LLM_WEIGHT, REAL_WEIGHT = 1.0, 4.0

# Recipe-scraper boilerplate that isn't a grocery item -- dropped, not trained.
_BOILER = re.compile(
    r"^(directions?|instructions?|serves?|to serve|for serving|serve with|servings?|"
    r"yields?|original recipe yields|ingredient checklist|get ingredients|prep time|"
    r"cook(ing)? time|total time|calories|kcal|nutrition|net carbs?|us customary|metric)\b",
    re.I)


def _is_val(key):
    # deterministic ~14% val fold, disjoint from the test bucket
    return int(hashlib.md5((key + "|val").encode("utf-8")).hexdigest(), 16) % 7 == 0

# Multilingual state words to synthesize training examples so the LEARNED state
# head recognizes frozen/canned in every language (English recipe data alone has
# almost none). (word, position) -- pre = before item, post = after.
FROZEN_AUG = [("frozen","pre"),("tiefkühl","pre"),("tiefgekühlt","pre"),("surgelé","pre"),
              ("surgelées","post"),("congelado","post"),("congelados","post"),("congelato","post"),
              ("diepvries","pre"),("mrożony","pre"),("fryst","pre"),("pakaste","pre"),
              ("замороженный","pre"),("冷凍","pre"),("congelat","post"),("gefroren","post")]
CANNED_AUG = [("canned","pre"),("tinned","pre"),("1 can","pre"),("2 cans","pre"),("jar","pre"),
              ("en conserve","post"),("en lata","post"),("enlatado","post"),("in scatola","post"),
              ("konserven","post"),("dose","pre"),("blik","pre"),("puszka","pre"),
              ("säilyke","pre"),("консервированный","pre"),("缶詰","post"),("罐装","pre")]
STATE_AUG_ITEMS = 5000   # food items to synthesize state variants from
AUG_WEIGHT = 1.0


def state_augment(data, rng):
    # sample FOOD items (skip nonfood) to attach multilingual state words to
    food = [(mt, v[0]) for mt, v in data.items() if v[0] != "nonfood" and v[1] == "none"]
    rng.shuffle(food)
    added = 0
    for mt, base in food[:STATE_AUG_ITEMS]:
        for words, state in ((FROZEN_AUG, "frozen"), (CANNED_AUG, "canned")):
            w, pos = rng.choice(words)
            aug = f"{w} {mt}" if pos == "pre" else f"{mt} {w}"
            if aug not in data:
                data[aug] = (base, state, AUG_WEIGHT)
                added += 1
    return added


def main():
    test_items = set(l.strip() for l in open("test_items.txt", encoding="utf-8") if l.strip())
    base_of = {}
    for line in open("base_labels.tsv", encoding="utf-8"):
        if "\t" not in line:
            continue
        k, b = line.rstrip("\n").split("\t", 1)
        if b.strip() in BASE_AISLES:
            base_of[k] = b.strip()

    # real overrides: hold out a val fold (for checkpoint selection), fold the rest
    # into training. ALL aisles now, not just CLEAR.
    overrides = []
    try:
        for line in open("real_trainfold.tsv", encoding="utf-8"):
            if "\t" not in line:
                continue
            raw, aisle = line.rstrip("\n").split("\t", 1)
            aisle = aisle.strip()
            key = light_normalize(raw)
            if aisle in FINAL_AISLES and key and key not in test_items:
                overrides.append((raw, aisle, key))
    except FileNotFoundError:
        pass
    val_keys = {key for _, _, key in overrides if _is_val(key)}

    # model_text -> (base, state, weight). LLM (recipe data) first; exclude test+val.
    data = {}
    n_llm = 0
    for line in open("data.txt", encoding="utf-8"):
        if "____" not in line:
            continue
        raw = line.split("____", 1)[1]
        key = light_normalize(raw)
        if not key or key in test_items or key in val_keys:
            continue
        base = base_of.get(key)
        if base is None:
            continue
        mt = normalize_model(raw)
        if len(mt) < 2 or _BOILER.match(mt):
            continue
        if mt not in data:
            data[mt] = (base, detect_state(mt), LLM_WEIGHT)
            n_llm += 1

    # overlay the train-fold overrides
    n_base = n_state = n_skip = 0
    val_rows = []
    for raw, aisle, key in overrides:
        if _is_val(key):
            val_rows.append((aisle, raw))
            continue
        mt = normalize_model(raw)
        if len(mt) < 2:
            continue
        if aisle in BASE_AISLES:                     # user filed under a base aisle -> state none
            data[mt] = (aisle, "none", REAL_WEIGHT)
            n_base += 1
        else:                                        # frozen/canned -> state target; base from labeler
            base = base_of.get(key)
            if base is None:
                n_skip += 1
                continue
            data[mt] = (base, aisle, REAL_WEIGHT)
            n_state += 1

    with open("real_val.tsv", "w", encoding="utf-8") as f:
        f.write("# real-override validation fold (final-aisle truth) for checkpoint selection\n")
        for aisle, raw in val_rows:
            f.write(f"{aisle}____{raw}\n")

    n_aug = state_augment(data, random.Random(42))

    with open("train.tsv", "w", encoding="utf-8") as f:
        for mt, (base, state, w) in data.items():
            f.write(f"{mt}\t{base}\t{state}\t{w}\n")

    bdist = collections.Counter(v[0] for v in data.values())
    sdist = collections.Counter(v[1] for v in data.values())
    print(f"LLM rows: {n_llm:,}  real base: {n_base:,}  real frozen/canned: {n_state:,}"
          f"  (skipped f/c w/o base: {n_skip})  val fold: {len(val_rows):,}"
          f"  state-aug: {n_aug:,}  total: {len(data):,}")
    print("base dist:", dict(bdist.most_common()))
    print("state dist:", dict(sdist))


if __name__ == "__main__":
    main()
