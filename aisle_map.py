"""
Single source of truth for the taxonomy: the 12 BASE aisles, the 2 STATE-derived
aisles (frozen/canned), the fuzzy PANTRY cluster (for the honest merged metric),
a cost matrix (cheap within pantry, expensive across food groups / state), and
the mapping from real user override category names -> final 14 aisles.
"""

BASE_AISLES = ["produce","dairy","meat","seafood","bakery","baking","spices",
               "grocery","condiments","beverages","liquor","nonfood"]
FINAL_AISLES = BASE_AISLES + ["frozen", "canned"]

# genuinely-interchangeable "center store" cluster -> merged for honest metric
PANTRY = {"grocery","spices","baking","condiments","bakery"}


def pantry_merge(a):
    return "PANTRY" if a in PANTRY else a


# --- user override category name -> final aisle (incl frozen/canned), or None to drop
_U = {
    # produce
    "produce":"produce","obst und gemüse":"produce","fruits & vegetables":"produce",
    "fruits et légumes":"produce","fruits et legumes":"produce","veg":"produce",
    "veggies":"produce","vegetables":"produce","fruit":"produce","groenten":"produce",
    "groenten/fruit":"produce","groente/fruit":"produce","frutas y verduras":"produce",
    "hedelmät & vihannekset":"produce","λαχανικά":"produce","warzywa":"produce",
    "légumes":"produce","frutas":"produce","fruits":"produce",
    # dairy
    "dairy":"dairy","milchprodukte":"dairy","produits laitiers":"dairy","cheese":"dairy",
    "zuivel":"dairy","maitotuotteet":"dairy","milch, eier & co":"dairy","butter":"dairy",
    "eggs":"dairy","dairy & eggs":"dairy","milch":"dairy",
    # meat
    "meat":"meat","protein":"meat","fleisch":"meat","viande":"meat","liha":"meat",
    "cooked/cured meat":"meat","cooked /cured meat":"meat","deli":"meat","poultry":"meat",
    # seafood
    "seafood":"seafood","fish":"seafood","fisch":"seafood","poisson":"seafood",
    "fish & seafood":"seafood","pescado":"seafood",
    # bakery (ready-to-eat)
    "bakery":"bakery","boulangerie":"bakery","bread":"bakery","backwaren":"bakery","brot":"bakery",
    # baking (ingredients)
    "baking":"baking","baking goods":"baking","baking supplies":"baking","flours":"baking",
    "flour":"baking","baking aisle":"baking","backen":"baking","baking/seasoning":"baking",
    # spices
    "spices":"spices","seasonings/spices":"spices","seasoning":"spices","seasonings":"spices",
    "gewürze":"spices","kruiden":"spices","herbs & spices":"spices","herbs and spices":"spices",
    "herbs":"spices","spice":"spices","μπαχαρικά":"spices","hierbas y especias":"spices",
    "herbs & spices - dried":"spices",
    # grocery
    "grocery":"grocery","pantry":"grocery","lebensmittel":"grocery","épicerie":"grocery",
    "dry goods":"grocery","snacks":"grocery","bulk":"grocery","pasta & rice":"grocery",
    "pasta":"grocery","beans":"grocery","dry beans":"grocery","nuts":"grocery",
    "nuts and seeds":"grocery","boodschappen":"grocery","international":"grocery",
    "snacks/chips":"grocery","cereal":"grocery","grains":"grocery","rice":"grocery",
    "quick meals":"grocery",
    # condiments
    "condiments":"condiments","oil & vinegars":"condiments","oils":"condiments",
    "oil":"condiments","sauces":"condiments","sauce":"condiments","sauce & spice":"condiments",
    "condiments, sauces, oils":"condiments","condimenten en sauzen":"condiments",
    "öl, mayo, ...":"condiments","vinegar":"condiments","dressings":"condiments",
    "syrup":"condiments","spreads":"condiments",
    # beverages
    "beverages":"beverages","boissons":"beverages","drinks":"beverages","getränke":"beverages",
    # liquor
    "liquor":"liquor","alcohol":"liquor","wine":"liquor","beer":"liquor","bières blondes":"liquor",
    # nonfood
    "nonfood":"nonfood","kosmetik":"nonfood","reinigungsmittel":"nonfood","cleaning":"nonfood",
    "vitamins":"nonfood","minerals":"nonfood","non alimentaire":"nonfood",
    "home improvement":"nonfood","pets":"nonfood","baby stuff":"nonfood","health":"nonfood",
    "household":"nonfood","home":"nonfood",
    # frozen
    "frozen":"frozen","tiefkühlware":"frozen","tiefkühl":"frozen","frozen foods":"frozen",
    "congelados":"frozen","surgelés":"frozen","diepvries":"frozen","pakaste":"frozen",
    # canned
    "canned":"canned","konserven":"canned","en conserve":"canned","säilykkeet":"canned",
    "conservas":"canned","tinned":"canned","canned goods":"canned","conserve":"canned",
}


def user_cat_to_aisle(cat):
    return _U.get((cat or "").strip().lower())


# cost of predicting `pred` when truth is `gold`. 0 correct; cheap within pantry;
# expensive across food groups or wrong state. Used for cost-weighted eval.
def cost(gold, pred):
    if gold == pred:
        return 0.0
    if gold in PANTRY and pred in PANTRY:
        return 0.3            # interchangeable center-store bins
    if {gold, pred} & {"frozen", "canned"}:
        return 1.5            # wrong aisle section, shopper walks elsewhere
    return 1.0
