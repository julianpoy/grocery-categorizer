"""
Factorized categorizer: a fine-tuned multilingual encoder (set via ENCODER)
with TWO heads:
  - base aisle (12 classes)   : semantic food type, state-agnostic
  - state (3 classes)         : none / frozen / canned, learned from text

Inference composes them: state!=none overrides to the frozen/canned aisle,
else the base aisle - so frozen/canned is a learned, multilingual signal.

Trains on train.tsv (item, base, state, weight). Saves ./aisle_model/.
"""
import json, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from transformers import get_cosine_schedule_with_warmup

from aisle_map import BASE_AISLES, pantry_merge
from state_utils import STATES
from text_norm import normalize_model

MODEL_NAME = os.environ.get("ENCODER", "google/embeddinggemma-300m")
PREFIX = os.environ.get("PREFIX", "query: ")   # input prefix; override per-encoder
DATA = "train.tsv"
OUT = Path(os.environ.get("OUTDIR", "./aisle_model"))
EPOCHS = int(os.environ.get("EPOCHS", "4"))
BATCH = 256
LR = 2e-5
STATE_LOSS_W = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base2id = {a: i for i, a in enumerate(BASE_AISLES)}
state2id = {s: i for i, s in enumerate(STATES)}


class DS(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return self.rows[i]


def collate(batch):
    texts = [PREFIX + r[0] for r in batch]
    b = torch.tensor([r[1] for r in batch])
    s = torch.tensor([r[2] for r in batch])
    w = torch.tensor([r[3] for r in batch], dtype=torch.float)
    return texts, b, s, w


class Heads(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(dim, 384), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(384, len(BASE_AISLES)))
        self.state = nn.Sequential(nn.Linear(dim, 96), nn.ReLU(),
                                   nn.Linear(96, len(STATES)))
    def forward(self, emb):
        return self.base(emb), self.state(emb)


def load_rows():
    rows = []
    for line in open(DATA, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) != 4:
            continue
        item, base, state, w = p
        if base in base2id and state in state2id:
            rows.append((item, base2id[base], state2id[state], float(w)))
    return rows


def class_weights(ids, n):
    c = np.bincount(ids, minlength=n).astype(float)
    c[c == 0] = 1
    w = c.sum() / (n * c)
    w = np.sqrt(w)              # soften
    return torch.tensor(w / w.mean(), dtype=torch.float, device=DEVICE)


def load_real_val(path):
    pairs = []
    if not os.path.exists(path):
        return pairs
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "____" not in line:
            continue
        a, t = line.split("____", 1)
        pairs.append((a.strip(), t.strip()))
    return pairs


def main():
    OUT.mkdir(exist_ok=True)
    rows = load_rows()
    print(f"training rows: {len(rows):,}")
    tr, va = train_test_split(rows, test_size=0.05, random_state=42,
                              stratify=[r[1] for r in rows])

    json.dump({"base2id": base2id, "state2id": state2id},
              open(OUT / "label_mapping.json", "w"), indent=2)

    enc = SentenceTransformer(MODEL_NAME, trust_remote_code=True).to(DEVICE)
    dim = enc.get_sentence_embedding_dimension()
    heads = Heads(dim).to(DEVICE)

    w_base = class_weights([r[1] for r in tr], len(BASE_AISLES))
    w_state = class_weights([r[2] for r in tr], len(STATES))
    print("state class weights:", {STATES[i]: round(w_state[i].item(), 2) for i in range(3)})
    ce_base = nn.CrossEntropyLoss(weight=w_base, reduction="none")
    ce_state = nn.CrossEntropyLoss(weight=w_state, reduction="none")

    opt = torch.optim.AdamW(list(enc.parameters()) + list(heads.parameters()), lr=LR)
    tl = DataLoader(DS(tr), batch_size=BATCH, shuffle=True, collate_fn=collate)
    vl = DataLoader(DS(va), batch_size=BATCH, collate_fn=collate)
    total_steps = len(tl) * EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    # Real user-override fold -> checkpoint on pantry-merged (the production metric),
    # not on in-distribution base accuracy. Falls back to base acc if absent.
    real_val = load_real_val("real_val.tsv")
    if real_val:
        print(f"real-override val fold: {len(real_val)} items (selecting on pantry-merged)")

    def real_val_merged():
        texts = [PREFIX + (normalize_model(t) or t) for _, t in real_val]
        with torch.no_grad():
            emb = enc.encode(texts, convert_to_tensor=True, show_progress_bar=False, batch_size=128)
            lb, ls = heads(emb)
        bi, si = lb.argmax(1).tolist(), ls.argmax(1).tolist()
        ok = 0
        for (g, _), b, s in zip(real_val, bi, si):
            st = STATES[s]
            pred = "frozen" if st == "frozen" else "canned" if st == "canned" else BASE_AISLES[b]
            ok += pantry_merge(pred) == pantry_merge(g)
        return ok / len(real_val)

    best, sel_name = -1.0, "val_base_acc"
    for ep in range(EPOCHS):
        enc.train(); heads.train()
        tot = 0
        for texts, b, s, w in tl:
            opt.zero_grad()
            feats = enc.tokenize(texts)
            feats = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in feats.items()}
            emb = enc.forward(feats)["sentence_embedding"]
            lb, ls = heads(emb)
            b, s, w = b.to(DEVICE), s.to(DEVICE), w.to(DEVICE)
            loss = (ce_base(lb, b) * w).mean() + STATE_LOSS_W * (ce_state(ls, s) * w).mean()
            loss.backward(); opt.step(); sched.step()
            tot += loss.item()

        enc.eval(); heads.eval()
        correct_b = correct_s = total = 0
        with torch.no_grad():
            for texts, b, s, w in vl:
                emb = enc.encode(texts, convert_to_tensor=True, show_progress_bar=False)
                lb, ls = heads(emb)
                correct_b += (lb.argmax(1).cpu() == b).sum().item()
                correct_s += (ls.argmax(1).cpu() == s).sum().item()
                total += len(b)
        ab, as_ = correct_b / total, correct_s / total
        if real_val:
            sel, sel_name = real_val_merged(), "real_val_merged"
        else:
            sel, sel_name = ab, "val_base_acc"
        print(f"epoch {ep+1}/{EPOCHS} loss={tot/len(tl):.4f} val_base_acc={ab:.4f} "
              f"val_state_acc={as_:.4f}" + (f" {sel_name}={sel:.4f}" if real_val else ""))
        if sel > best:
            best = sel
            enc.save(str(OUT / "sentence_transformer"))
            torch.save(heads.state_dict(), OUT / "heads.pt")
            print(f"  saved ({sel_name}={sel:.4f})")

    print(f"\ndone. best {sel_name}={best:.4f}. model in {OUT}")


if __name__ == "__main__":
    main()
