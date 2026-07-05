"""
Production inference for the factorized aisle model - torch-free.

Serves the int8 ONNX artifact built by `export_onnx.py`. Dependencies at run
time are only: onnxruntime, tokenizers, numpy (see requirements-production.txt).
No torch / transformers / sentence-transformers.

Pipeline (mirrors the trained SentenceTransformer exactly):
    tokenize -> Gemma3 transformer (ONNX int8) -> mean-pool -> Dense(folded)
    -> L2-normalize -> {base head, state head} -> compose.

    from inference_onnx import AisleCategorizer
    cat = AisleCategorizer()
    cat.predict_batch(["2 lbs chicken breast", "1 can black beans", "leche entera"])
"""
import os
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from text_norm import normalize_model

DEFAULT_DIR = os.environ.get("AISLE_MODEL_DIR", "aisle_model_onnx")
# CPU serving: cap threads. Un-capped, onnxruntime spawns one thread per core,
# which *slows* single short-sequence requests dramatically on many-core hosts.
DEFAULT_THREADS = int(os.environ.get("OMP_NUM_THREADS", "4"))


class AisleCategorizer:
    def __init__(self, model_dir=DEFAULT_DIR, num_threads=DEFAULT_THREADS):
        d = Path(model_dir)
        self.meta = json.load(open(d / "meta.json"))
        self.prefix = self.meta["prefix"]
        self.base_aisles = self.meta["base_aisles"]
        self.states = self.meta["states"]
        self.uncertain = float(self.meta["uncertain"])
        self.max_seq = int(self.meta["max_seq"])

        self.tok = Tokenizer.from_file(str(d / "tokenizer" / "tokenizer.json"))
        self.tok.enable_truncation(max_length=self.max_seq)
        pad_id = self.tok.token_to_id("<pad>")
        pad_id = 0 if pad_id is None else pad_id
        self.tok.enable_padding(pad_id=pad_id, pad_token="<pad>")

        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(d / "encoder_int8.onnx"), sess_options=so,
            providers=["CPUExecutionProvider"])

        t = np.load(d / "tail.npz")
        self.dense_W = t["dense_W"]
        self.bw0, self.bb0, self.bw3, self.bb3 = t["base_w0"], t["base_b0"], t["base_w3"], t["base_b3"]
        self.sw0, self.sb0, self.sw2, self.sb2 = t["state_w0"], t["state_b0"], t["state_w2"], t["state_b2"]

    def _embed(self, texts):
        encs = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        tok_emb = self.sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        m = mask[:, :, None].astype(np.float32)
        pooled = (tok_emb * m).sum(1) / np.clip(m.sum(1), 1e-9, None)   # mean pool
        proj = pooled @ self.dense_W.T                                 # folded Dense
        proj /= np.clip(np.linalg.norm(proj, axis=1, keepdims=True), 1e-12, None)  # L2
        return proj.astype(np.float32)

    @staticmethod
    def _softmax(x):
        x = x - x.max(1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(1, keepdims=True)

    def predict_batch(self, items, batch_size=64):
        out = []
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            texts = [self.prefix + (normalize_model(t) or t) for t in chunk]
            emb = self._embed(texts)
            hb = np.maximum(emb @ self.bw0.T + self.bb0, 0) @ self.bw3.T + self.bb3
            hs = np.maximum(emb @ self.sw0.T + self.sb0, 0) @ self.sw2.T + self.sb2
            pb, ps = self._softmax(hb), self._softmax(hs)
            for j, item in enumerate(chunk):
                bi, si = int(pb[j].argmax()), int(ps[j].argmax())
                base, state = self.base_aisles[bi], self.states[si]
                bconf, sconf = float(pb[j, bi]), float(ps[j, si])
                if state == "frozen":
                    cat, conf = "frozen", sconf
                elif state == "canned":
                    cat, conf = "canned", sconf
                else:
                    cat, conf = base, bconf
                out.append({
                    "item": item, "category": cat, "confidence": round(conf, 4),
                    "base": base, "state": state, "uncertain": conf < self.uncertain,
                })
        return out

    def predict(self, item):
        return self.predict_batch([item])[0]


if __name__ == "__main__":
    cat = AisleCategorizer()
    tests = ["2 lbs chicken breast", "1 can black beans", "frozen peas", "leche entera",
             "aceite de oliva virgen extra", "tiefkühl pizza", "all-purpose flour",
             "toilet paper", "cheddar cheese", "红酒", "2 jars marinara sauce",
             "fresh atlantic salmon", "surgelé crevettes"]
    for r in cat.predict_batch(tests):
        flag = " ??" if r["uncertain"] else ""
        print(f"  {r['item']:32s} -> {r['category']:10s} "
              f"(base={r['base']}, state={r['state']}, {r['confidence']:.2f}){flag}")
