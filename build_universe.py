"""
Build the universe of unique items to be labeled by the LLM.

Uses light, model-appropriate normalization (from text_norm): strip leading
quantities, units, prices and list markers; collapse whitespace; lowercase --
but KEEP meaningful words like frozen/canned/dried/fresh, which carry aisle
signal the model should learn from.

Sources: data.txt plus the real user-override titles. Output: universe.txt.
"""
import csv
import os
from pathlib import Path
from collections import Counter

# Normalization lives in text_norm (stdlib-only); re-exported here for callers
# that do `from build_universe import normalize_model`.
from text_norm import (  # noqa: F401
    UNIT_WORDS, STATE_UNITS, _normalize, light_normalize, normalize_model,
)


def main():
    counts = Counter()
    with open("data.txt", encoding="utf-8") as f:
        for line in f:
            if "____" not in line:
                continue
            _, text = line.split("____", 1)
            n = light_normalize(text)
            if len(n) >= 2:
                counts[n] += 1
    n_recipe = len(counts)

    src = Path(os.environ.get("USER_CATEGORIZATIONS", "user-categorizations.csv"))
    if src.exists():
        for r in csv.DictReader(open(src, encoding="utf-8")):
            n = light_normalize(r.get("title", ""))
            if len(n) >= 2:
                counts[n] += 1

    items = [w for w, _ in counts.most_common()]
    with open("universe.txt", "w", encoding="utf-8") as f:
        for it in items:
            f.write(it + "\n")

    # crude script/language mix
    def script(s):
        for ch in s:
            o = ord(ch)
            if 0x4E00 <= o <= 0x9FFF: return "CJK"
            if 0x3040 <= o <= 0x30FF: return "JP-kana"
            if 0x0400 <= o <= 0x04FF: return "Cyrillic"
            if 0x0370 <= o <= 0x03FF: return "Greek"
            if 0x0590 <= o <= 0x05FF: return "Hebrew"
            if 0x0600 <= o <= 0x06FF: return "Arabic"
        return "Latin/other"
    sc = Counter(script(it) for it in items)

    print(f"Unique items from recipe data : {n_recipe:,}")
    print(f"Total unique items (+overrides): {len(items):,}")
    print(f"\nScript mix (non-Latin = multilingual signal):")
    for s, n in sc.most_common():
        print(f"  {s:14s} {n:6,} ({n/len(items)*100:.1f}%)")
    print(f"\nSample items:")
    for it in items[:15]:
        print(f"  {it}")
    print(f"\nWrote universe.txt")


if __name__ == "__main__":
    main()
