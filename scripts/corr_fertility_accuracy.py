"""D4: Spearman correlation of fertility vs baseline (condition=full) accuracy across languages.

Fertility is taken from each model's OWN tokenizer. Caveat for the report: resource level confounds
this correlation (the reason E3-E4 exist), and n = 7 languages gives very little power.
Writes results/corr_fertility_accuracy.csv.
"""
import os

import pandas as pd
from scipy.stats import spearmanr

from tokentax.metrics import summarize_e1
from tokentax.schemas import load

MODEL_TOKENIZER = {"qwen2.5-3b": "qwen2.5", "gemma2-2b": "gemma2"}

s = summarize_e1(pd.read_csv("results/e1_intrinsic.csv"))
e3 = load("e3_results")
acc = e3[e3.condition == "full"].groupby(["model", "lang"], as_index=False).correct.mean()

rows = []
for m, g in acc.groupby("model"):
    tok = MODEL_TOKENIZER.get(m)
    st = s[s.tokenizer == tok].set_index("lang")
    g = g[g.lang.isin(st.index)]
    for metric in ["fertility", "tax_vs_eng"]:
        rho, p = spearmanr(g.lang.map(st[metric]), g.correct)
        rows.append({"model": m, "tokenizer": tok, "metric": metric, "n_langs": len(g),
                     "spearman_rho": round(rho, 3), "p_value": round(p, 4)})
out = pd.DataFrame(rows)
dest = "results" if os.path.exists("results/e3_results.csv") else "results/fake"  # never mix fake into results/
out.to_csv(f"{dest}/corr_fertility_accuracy.csv", index=False)
print(f"wrote {dest}/corr_fertility_accuracy.csv")
print(out.to_string(index=False))
