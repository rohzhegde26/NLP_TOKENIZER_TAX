"""Cost bar chart: INR per 1k English-equivalent words, per language, for priced tokenizers."""
import matplotlib.pyplot as plt
import numpy as np

from tokentax import plotstyle as ps
from tokentax.schemas import load

ps.apply()
c = load("e2_cost").dropna(subset=["inr_per_1k_eng_words"])
toks = list(dict.fromkeys(c.tokenizer))
langs = ps.ordered(c.lang)
w = 0.8 / max(len(toks), 1)
fig, ax = plt.subplots(figsize=(10, 4))
for k, t in enumerate(toks):
    v = c[c.tokenizer == t].set_index("lang").reindex(langs).inr_per_1k_eng_words
    ax.bar(np.arange(len(langs)) + k * w - 0.4 + w / 2, v, w, label=ps.TOKENIZER_LABELS.get(t, t))
ax.set_xticks(range(len(langs)), [ps.lang_label(l) for l in langs], rotation=35, ha="right")
ax.set_ylabel("INR per 1,000 English-equivalent words")
ax.set_title(f"Input cost for the same content (prices {c.price_snapshot_date.iloc[0]})")
ax.legend(frameon=False, fontsize=8)
ps.save(fig, "fig_cost")
