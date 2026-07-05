"""
Multilingual preservation-STATE detection: frozen / canned / none.

Two uses:
  1. Bootstrap training TARGETS for the learned state head (the model then
     generalizes past this keyword list to unseen phrasings/misspellings).
  2. Sanity fallback only; at serve time the STATE HEAD predicts state.

Keywords span the ~20 supported languages. Matched as whole words (Latin) or
substrings (CJK/no word boundaries). Frozen takes precedence over canned.
"""
import re, unicodedata

# whole-word (Latin-ish) keywords
FROZEN_WORDS = {
    "frozen", "freeze",                      # en
    "congelado","congelada","congelados","congeladas",  # es/pt
    "surgelé","surgele","surgelée","congelé","congele","congelée","surgelati","surgelato","surgelata",  # fr/it
    "tiefkühl","tiefgekühlt","tiefkuehl","gefroren","tk",  # de
    "congelato","congelati","congelata",     # it
    "diepvries","bevroren",                  # nl
    "mrożony","mrożone","mrożona","mrozony", # pl
    "fryst","frysta","frusen",               # sv
    "frossen","frost","frosne",              # da/no
    "pakaste","pakastettu",                  # fi
    "congelat","congelate","congelata",      # ro
    "šaldyta","saldyta","šaldytas",          # lt
    "izoztu","izoztutako",                   # eu
    "fagyasztott",                           # hu
}
CANNED_WORDS = {
    "canned","tinned","tin","tins","can","cans","jar","jars","jarred",  # en
    "enlatado","enlatada","lata","conserva","conservas",  # es/pt
    "conserve","boîte","boite","boites","conserva",       # fr/it
    "dose","dosen","konserve","konserven","eingemacht","eingelegt","glas",  # de
    "scatola","lattina","lattine",           # it
    "blik","ingeblikt","conserven",          # nl
    "puszka","puszce","konserwa",            # pl
    "konserv","burk","burkar",               # sv
    "dåse","daase","konserves",              # da
    "säilyke","säilykkeet","purkki","tölkki",# fi
    "conservă","conserva","cutie",           # ro
    "konzerv",                               # hu
    "konservuota","konservai",               # lt
}
# substring keywords for scripts without spaces / non-Latin
FROZEN_SUB = ["冷凍","冷冻","냉동","заморож","заморож","κατεψυγμ","קפוא","مجمد","мороже"]
CANNED_SUB = ["缶詰","罐装","罐頭","罐头","консерв","банк","κονσέρβ","שימור","معلب","консерв"]

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_state(text: str):
    if not text:
        return "none"
    t = unicodedata.normalize("NFKC", text).lower()
    # substring (non-Latin scripts)
    if any(s in text for s in FROZEN_SUB):
        return "frozen"
    if any(s in text for s in CANNED_SUB):
        return "canned"
    toks = set(_WORD.findall(t))
    if toks & FROZEN_WORDS:
        return "frozen"
    if toks & CANNED_WORDS:
        return "canned"
    return "none"


STATES = ["none", "frozen", "canned"]

if __name__ == "__main__":
    for s in ["frozen peas","canned black beans","surgelé poulet","tiefkühl pizza",
              "guisantes congelados","缶詰のトマト","замороженная рыба","fresh milk",
              "1 jar marinara","säilyke tonnikala","mrożony groszek"]:
        print(f"{s:28s} -> {detect_state(s)}")
