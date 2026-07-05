"""
Torch-free text normalization shared by the data-build pipeline and the
production server. `normalize_model` is the MODEL INPUT form: it strips
quantities/units/prices/list-markers but KEEPS meaningful words
(frozen/canned/fresh/can/jar) that carry aisle + state signal.

This is intentionally dependency-light (stdlib only) so the serving image does
not need torch, transformers, or the data-build scripts.
"""
import re
import unicodedata

UNIT_WORDS = {
    "cup", "cups", "c", "tbsp", "tbs", "tablespoon", "tablespoons", "tsp", "teaspoon",
    "teaspoons", "oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds", "g", "gr", "gram",
    "grams", "kg", "mg", "ml", "l", "liter", "liters", "litre", "litres", "pint", "pints",
    "quart", "quarts", "gallon", "gallons", "clove", "cloves", "sprig", "sprigs", "bunch",
    "pinch", "dash", "slice", "slices", "stick", "sticks", "head", "stalk", "stalks",
    "piece", "pieces", "can", "cans", "jar", "jars", "package", "packages", "pkg", "pack",
    "packs", "bag", "bags", "box", "boxes", "bottle", "bottles", "container", "carton",
    "cartons", "dozen", "small", "medium", "large", "x", "lb.", "oz.", "g.", "kg.", "ml.",
}
# Packaging words that double as the canned-aisle cue. Kept in the model input
# so the state head can learn "1 can beans" -> canned.
STATE_UNITS = {"can", "cans", "jar", "jars"}

_PRICE = re.compile(r"\$\s*\d+(?:[.,]\d+)?")
_LEAD = re.compile(r"^[\s•\*\-–—;:.,#>]+")
_NUM = re.compile(r"\d+(?:[.,/]\d+)?")
_FRAC = "½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅐⅑⅒"


def _normalize(text: str, drop_state_units: bool) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text).lower().strip()
    t = _PRICE.sub(" ", t)
    t = _LEAD.sub("", t)
    t = "".join(" " if ch in _FRAC else ch for ch in t)
    t = _NUM.sub(" ", t)
    drop = UNIT_WORDS if drop_state_units else (UNIT_WORDS - STATE_UNITS)
    toks = [w for w in re.split(r"\s+", t) if w and w not in drop]
    t = " ".join(toks)
    t = re.sub(r"\s+", " ", t).strip(" ,.-;:()[]")
    return t.strip()


# Leading measurement-fragment words left behind by scraped recipe quantities
# (e.g. "(14 oz)" -> "oz)"). Stripped from the model input only.
_MEAS_LEAD = {"of", "inch", "thick", "ct", "pkg", "pkgs"}


def _strip_fragments(t: str) -> str:
    """Drop leading recipe-quantity fragments from the MODEL INPUT: orphan `oz)`,
    `ml)`, fraction slashes, `of`, stray single chars. Real shopping-list input
    has none of these, so this is a no-op on clean input and only de-noises the
    scraped recipe training data (keeping train/serve text consistent)."""
    if not t:
        return t
    t = t.replace("⁄", " ")
    toks = t.split()
    while toks:
        w = toks[0]
        core = w.strip(").,;:")
        if w.endswith(")") or core in _MEAS_LEAD or len(core) <= 1:
            toks.pop(0)
            continue
        break
    return re.sub(r"\s+", " ", " ".join(toks)).strip(" ,.-;:()[]")


def light_normalize(text: str) -> str:
    """Cue-STRIPPING form: base-label lookup key (drops can/jar). Unchanged so
    existing LLM labels (keyed by this form) keep matching."""
    return _normalize(text, drop_state_units=True)


def normalize_model(text: str) -> str:
    """Cue-KEEPING form: the MODEL INPUT (retains can/jar/frozen/canned cues),
    with leading recipe-quantity fragments stripped."""
    return _strip_fragments(_normalize(text, drop_state_units=False))
