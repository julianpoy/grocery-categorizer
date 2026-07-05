"""
Build a torch-free, int8 serving artifact from a fine-tuned factorized model.

Takes a trained ./aisle_model-style dir (SentenceTransformer encoder + heads.pt)
and produces an `aisle_model_onnx/` bundle that serves with ONLY
onnxruntime + tokenizers + numpy (no torch, transformers, or sentence-transformers):

    aisle_model_onnx/
      encoder_int8.onnx     quantized Gemma3 transformer (input_ids,attention_mask -> token_emb)
      tail.npz              folded Dense projection + both classifier heads (fp32, tiny)
      tokenizer/            tokenizer.json (+ config) for the `tokenizers` lib
      meta.json             prefix, aisle/state label order, pooling, threshold, dims

Usage:  SRC=aisle_model OUT=aisle_model_onnx python export_onnx.py
"""
import os, json, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # export on CPU: avoids device-mismatch in torch.export
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from aisle_map import BASE_AISLES, FINAL_AISLES
from state_utils import STATES
from train_factorized import Heads, PREFIX

SRC = Path(os.environ.get("SRC", "aisle_model"))
OUT = Path(os.environ.get("OUT", "aisle_model_onnx"))
MAX_SEQ = int(os.environ.get("MAX_SEQ", "64"))
ST_DIR = SRC / "sentence_transformer"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = SentenceTransformer(str(ST_DIR), trust_remote_code=True).eval()

    # --- 1. export the transformer (module 0) to ONNX ------------------------
    auto = st[0].auto_model.eval()

    class Enc(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    fp32_path = OUT / "encoder_fp32.onnx"
    int8_path = OUT / "encoder_int8.onnx"
    ids = torch.randint(0, 5000, (2, 12))
    mask = torch.ones(2, 12, dtype=torch.long)
    torch.onnx.export(
        Enc(auto), (ids, mask), str(fp32_path),
        input_names=["input_ids", "attention_mask"], output_names=["token_emb"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                      "token_emb": {0: "b", 1: "s"}},
        opset_version=20,
    )
    print(f"exported fp32 ONNX ({fp32_path.stat().st_size/1e6:.0f} MB)")

    # --- 2. dynamic int8 quantization ---------------------------------------
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = OUT / "encoder_pre.onnx"
    quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True)
    quantize_dynamic(str(pre), str(int8_path), weight_type=QuantType.QInt8)
    print(f"quantized int8 ONNX ({int8_path.stat().st_size/1e6:.0f} MB)")
    # remove fp32 + pre-process intermediates, including ONNX external-data (.data) files
    for stem in ("encoder_fp32", "encoder_pre"):
        for junk in OUT.glob(stem + "*"):
            junk.unlink(missing_ok=True)

    # --- 3. fold the two Dense layers (no bias, Identity act) into one matmul -
    def dense_weight(mod):
        for p_name, p in mod.named_parameters():
            if p.ndim == 2:
                return p.detach().cpu().numpy()
        raise RuntimeError("no 2D weight in Dense module")
    W2 = dense_weight(st[2])           # [3072, 768]
    W3 = dense_weight(st[3])           # [768, 3072]
    Wc = (W3 @ W2).astype(np.float32)  # y = pooled @ Wc.T  ==  Dense3(Dense2(pooled))
    assert Wc.shape == (768, 768), Wc.shape

    # --- 4. extract the two classifier heads --------------------------------
    heads = Heads(st.get_sentence_embedding_dimension())
    heads.load_state_dict(torch.load(SRC / "heads.pt", map_location="cpu"))
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in heads.state_dict().items()}

    np.savez(
        OUT / "tail.npz",
        dense_W=Wc,
        base_w0=sd["base.0.weight"], base_b0=sd["base.0.bias"],
        base_w3=sd["base.3.weight"], base_b3=sd["base.3.bias"],
        state_w0=sd["state.0.weight"], state_b0=sd["state.0.bias"],
        state_w2=sd["state.2.weight"], state_b2=sd["state.2.bias"],
    )

    # --- 5. tokenizer + metadata --------------------------------------------
    tok_dir = OUT / "tokenizer"; tok_dir.mkdir(exist_ok=True)
    for fn in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = ST_DIR / fn
        if src.exists():
            shutil.copy(src, tok_dir / fn)

    json.dump({
        "prefix": PREFIX,
        "pooling": "mean",
        "base_aisles": BASE_AISLES,
        "states": STATES,
        "final_aisles": FINAL_AISLES,
        "emb_dim": st.get_sentence_embedding_dimension(),
        "max_seq": MAX_SEQ,
        "uncertain": 0.45,
        "encoder": "google/embeddinggemma-300m (fine-tuned, int8 ONNX)",
    }, open(OUT / "meta.json", "w"), indent=2)

    # Gemma license notice must travel with the model bundle (it's a Model Derivative).
    notice = Path("NOTICE")
    if notice.exists():
        shutil.copy(notice, OUT / "NOTICE")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"artifact written to {OUT}/  (total {total:.0f} MB)")


if __name__ == "__main__":
    main()
