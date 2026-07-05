# Grocery Aisle Categorizer

A multilingual API that sorts shopping-list items into **store aisles** for a recipe app. Input is short, messy, any-language item text (`"2 lbs chicken breast"`, `"1 can black beans"`, `"leche entera"`, `"tiefkühl pizza"`); output is one of 14 aisles.

## Approach: a factorized, model-driven classifier

Categorization is done by a **fine-tuned multilingual model**, not a lookup map. The model has two heads that are composed by one deterministic rule:

- **Base-aisle head** (12 classes) - the item's semantic food type, ignoring preservation state: `tomato → produce`, always.
- **State head** (3 classes: none / frozen / canned) - *learned from the text in any language* (`frozen`, `surgelé`, `tiefkühl`, `冷凍`, `1 can`, `en conserve`…).
- **Compose**: `state != none` → the frozen/canned aisle; else the base aisle. (This one rule encodes the store-shelving convention - canned tuna is in the canned aisle, not at the fresh fish counter.)

Encoder: **`google/embeddinggemma-300m`**, fine-tuned end-to-end. Training: `train_factorized.py` (any encoder via `ENCODER=`).

**14 aisles:** produce, dairy, meat, seafood, bakery (ready-to-eat baked goods), baking (flour/sugar/leaveners), spices, grocery (pantry dry goods), condiments (oils/sauces/dressings), beverages, liquor, nonfood, + frozen, canned (from the state head).

## Serving: torch-free int8 ONNX

Production inference (`inference_onnx.py` + `server.py`) runs the fine-tuned model as an **int8 ONNX graph** and needs only `onnxruntime`, `tokenizers`, and `numpy` - **no torch, transformers, or sentence-transformers**. The Gemma3 transformer runs in ONNX Runtime; mean-pooling, the folded projection, L2-normalize, and both heads run in NumPy.

| | value |
|---|---|
| artifact on disk (`aisle_model_onnx/`) | ~330 MB (294 MB int8 encoder + 32 MB tokenizer + heads) |
| live server RSS | ~575 MB (peak ~740 MB) |
| CPU latency | ~3.7 ms/item (4 threads); batch of 14 ≈ 37 ms end-to-end |

> **CPU threads:** the server pins `OMP_NUM_THREADS=4` (and onnxruntime `intra_op_num_threads`). Un-capped, onnxruntime spawns one thread per core, which *slows* single short-sequence requests dramatically on many-core hosts. Override the env var for your instance.

## How the training data is built

Human labels in the raw data are ~1-in-5 self-contradictory, so labels are generated fresh:

1. **`build_universe.py`** - dedupe all items into `universe.txt` (~89k). Two normalizers (in `text_norm.py`): a cue-*stripping* form for label lookup keys, and a cue-*keeping* form for model input (so the state head can see `1 can`/`frozen`).
2. **`run_labeler.py`** - a local LLM (`google/gemma-4-12B-it`, on GPU) assigns each item its **base aisle** (state-agnostic, so far less ambiguous than a final aisle). Validated against real user behavior: ~88% pantry-merged agreement.
3. **`state_utils.py`** - bootstraps state targets from a multilingual frozen/canned keyword list; the model then generalizes past it.
4. **`build_test.py`** - holds out 25% of real production overrides as a test set the model never sees.
5. **`build_training.py`** → `train.tsv`; **`train_factorized.py`** → `./aisle_model/`.

## Measured quality (held-out real production overrides)

Evaluated on 1,147 real items that users manually re-filed (a hard, failure-biased set), via `eval_factorized.py`:

| metric | value |
|---|---|
| pantry-merged accuracy | **75.4%** |
| exact accuracy | 61.6% |
| mean shopping cost/item¹ | **0.340** |

int8 quantization is accuracy-neutral here (fp32 75.5% → int8 75.4%): ONNX Runtime keeps the token-embedding table in fp32 and quantizes only the matmuls.

¹ Cost matrix: within-pantry mistakes are cheap (adjacent bins), cross-food-group and frozen/canned mistakes are expensive.

## Running the API

```bash
python server.py --host 0.0.0.0 --port 8000
curl -X POST localhost:8000/categorize -H 'Content-Type: application/json' \
  -d '{"items": ["2 lbs chicken breast", "1 can black beans", "leche entera"]}'
```

Each result always returns a `category` (the predicted aisle), plus `confidence`, `base`, `state`, and `uncertain` (a boolean: confidence is below the threshold). `uncertain` is advisory metadata only - it does not gate or change the returned category; the caller may use or ignore it.

## Rebuild from scratch

```bash
python build_universe.py
python run_labeler.py --model google/gemma-4-12B-it --input universe.txt --out base_labels.tsv
python build_test.py
python build_training.py
ENCODER=google/embeddinggemma-300m python train_factorized.py   # -> ./aisle_model (GPU)
python eval_factorized.py real_test.txt                          # honest held-out eval
SRC=./aisle_model python export_onnx.py                          # -> ./aisle_model_onnx (int8, torch-free)
```

Training requires a GPU and the `.venv` (torch + sentence-transformers + transformers + onnx/onnxruntime). `google/embeddinggemma-300m` is a **gated** model - accept the license on HuggingFace and `hf auth login` first. Serving requires only `requirements-production.txt` (onnxruntime + tokenizers + numpy + FastAPI).

## Notes & limitations

- **Frozen/canned** are inherently under-determined from item text (users rarely write the state). The model returns the sensible base aisle plus a `state` field; the app can layer in user context.
- The eval set is failure-biased (only items users overrode), so these numbers are a **floor**, not live accuracy. The true north-star metric is live override-rate in an A/B test.
- Remaining accuracy gap is dominated by **label ceiling** (the LLM labeler and user-taxonomy inconsistency in the fuzzy pantry cluster), not encoder capacity.
- This model was created with the assistance of Opus 4.8 - I'm not a machine learning expert.

## License

**Code:** the application code in this repository is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0) - see [`LICENSE`](LICENSE). AGPL is a network-copyleft license: anyone you give access to the running service is entitled to the corresponding source.

**Model:** the **served model is a Gemma Model Derivative** and is governed *separately* by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) - the AGPL applies to the code, not the model weights. This applies because it is fine-tuned from `google/embeddinggemma-300m` and its training labels were generated by `google/gemma-4-12B-it`. See the [`NOTICE`](NOTICE) file, which is bundled into `aisle_model_onnx/` and the container image.

If you **distribute the model** (the repo, a container image, or a model archive), you must: include the `NOTICE`, provide recipients a copy of the Gemma Terms of Use, note that the files were modified, and pass the Gemma use restrictions through to downstream users.

If you **only serve it via API** (a "Hosted Service" under the terms), the NOTICE-file requirement is waived, but you must still incorporate the Gemma use restrictions into your API terms of service.

> This is not legal advice; have counsel review before commercial deployment.
