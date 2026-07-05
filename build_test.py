"""
Build the held-out TEST set from real user overrides, with 14-aisle truth.

Split is by ITEM (deterministic hash) -- the override CSV has no user id, so
per-user splitting isn't possible (documented limitation). Test items are
excluded from training (see build_training.py) to prevent leakage.

Outputs:
  real_test.txt        aisle____title   (held-out ~25%)
  real_trainfold.tsv   norm_item<TAB>aisle   (other ~75%, clear-aisle real items to optionally fold into training)
  test_items.txt       light-normalized test item strings (exclusion list for training)
"""
import csv, hashlib, collections, os
from pathlib import Path
from aisle_map import user_cat_to_aisle, FINAL_AISLES
from build_universe import light_normalize

SRC = Path(os.environ.get("USER_CATEGORIZATIONS", "user-categorizations.csv"))
TEST_FRAC_BUCKET = 0  # items with hash%4 == 0 -> test (~25%)


def bucket(s):
    h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
    return h % 4


def main():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    seen = {}
    for r in rows:
        title = (r.get("title") or "").strip()
        aisle = user_cat_to_aisle(r.get("categoryTitle", ""))
        if not title or aisle is None:
            continue
        norm = light_normalize(title)
        if len(norm) < 2:
            continue
        # keep first mapping per normalized item (dedupe)
        seen.setdefault(norm, (title, aisle))

    test, trainfold, test_norms = [], [], []
    for norm, (title, aisle) in seen.items():
        if bucket(norm) == TEST_FRAC_BUCKET:
            test.append((aisle, title))
            test_norms.append(norm)
        else:
            trainfold.append((title, aisle))   # raw title (build_training re-normalizes)

    with open("real_test.txt", "w", encoding="utf-8") as f:
        f.write("# held-out real user-override test set (14-aisle truth, split by item)\n")
        for aisle, title in test:
            f.write(f"{aisle}____{title}\n")
    with open("real_trainfold.tsv", "w", encoding="utf-8") as f:
        for norm, aisle in trainfold:
            f.write(f"{norm}\t{aisle}\n")
    with open("test_items.txt", "w", encoding="utf-8") as f:
        for n in test_norms:
            f.write(n + "\n")

    dt = collections.Counter(a for a, _ in test)
    print(f"unique override items: {len(seen)}")
    print(f"  test  : {len(test)}")
    print(f"  train : {len(trainfold)}")
    print("test aisle dist:")
    for a, n in dt.most_common():
        print(f"  {a:11s} {n}")


if __name__ == "__main__":
    main()
