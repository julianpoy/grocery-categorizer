"""
Evaluate the factorized model on the held-out real-override test set.

Compose: state!=none -> frozen/canned aisle, else base aisle.
Reports (per the review's requirements): exact accuracy, PANTRY-MERGED accuracy
(honest measure of real errors vs interchangeable bins), COST-WEIGHTED error,
per-aisle, and the confusion matrix. Optionally a crude script/language slice.

    python3 eval_factorized.py [real_test.txt]
"""
import sys, os, collections
import torch
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from train_factorized import Heads, PREFIX
from aisle_map import BASE_AISLES, FINAL_AISLES, pantry_merge, cost
from state_utils import STATES
from build_universe import normalize_model

MODEL = Path(os.environ.get("OUTDIR", "./aisle_model"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self):
        self.enc = SentenceTransformer(str(MODEL / "sentence_transformer"), trust_remote_code=True).to(DEVICE).eval()
        self.heads = Heads(self.enc.get_sentence_embedding_dimension()).to(DEVICE)
        self.heads.load_state_dict(torch.load(MODEL / "heads.pt", map_location=DEVICE))
        self.heads.eval()

    def predict(self, texts):
        norm = [PREFIX + (normalize_model(t) or t) for t in texts]
        with torch.no_grad():
            emb = self.enc.encode(norm, convert_to_tensor=True, show_progress_bar=False,
                                  batch_size=128)
            lb, ls = self.heads(emb)
            base = lb.argmax(1).cpu().tolist()
            state = ls.argmax(1).cpu().tolist()
        out = []
        for b, s in zip(base, state):
            st = STATES[s]
            out.append("frozen" if st == "frozen" else "canned" if st == "canned"
                       else BASE_AISLES[b])
        return out


def load(path):
    pairs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "____" not in line:
            continue
        a, t = line.split("____", 1)
        pairs.append((a.strip(), t.strip()))
    return pairs


def has_nonlatin(s):
    return any(ord(c) > 0x2000 and not c.isspace() for c in s)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "real_test.txt"
    gold = load(path)
    preds = Model().predict([t for _, t in gold])

    n = len(gold)
    exact = merged = 0
    tot_cost = 0.0
    by = collections.defaultdict(lambda: [0, 0])
    conf = collections.Counter()
    lang = {"latin": [0, 0], "nonlatin": [0, 0]}
    for (g, t), p in zip(gold, preds):
        ok = p == g
        exact += ok
        merged += pantry_merge(p) == pantry_merge(g)
        tot_cost += cost(g, p)
        by[g][0] += ok; by[g][1] += 1
        if not ok:
            conf[(g, p)] += 1
        k = "nonlatin" if has_nonlatin(t) else "latin"
        lang[k][0] += ok; lang[k][1] += 1

    print(f"\nTEST: {path}   n={n}")
    print(f"  exact          : {exact/n*100:.1f}%")
    print(f"  pantry-merged  : {merged/n*100:.1f}%   (honest: real cross-group errors)")
    print(f"  mean cost/item : {tot_cost/n:.3f}   (0=perfect; <.3 within-pantry, 1 cross, 1.5 state)")
    print("\nper-aisle (exact):")
    for a in sorted(by):
        c, t = by[a]
        print(f"  {a:11s} {c:4d}/{t:4d}  {c/t*100:5.1f}%")
    print("\nlanguage slice (exact):")
    for k, (c, t) in lang.items():
        if t:
            print(f"  {k:9s} {c:4d}/{t:4d}  {c/t*100:5.1f}%")
    print("\ntop confusions (gold -> pred):")
    for (g, p), k in conf.most_common(12):
        print(f"  {g:11s} -> {p:11s} {k}")


if __name__ == "__main__":
    main()
