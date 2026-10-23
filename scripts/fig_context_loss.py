"""Context-loss chart: % of a context window lost vs English, per language and tokenizer."""
import matplotlib.pyplot as plt

from tokentax import plotstyle as ps
from tokentax.schemas import load
from tokentax.tokenizers import TOKENIZERS

ps.apply()
c = load("e2_cost")
langs = [l for l in ps.ordered(c.lang) if l != "eng_Latn"]
fig, ax = plt.subplots(figsize=(10, 4))
for t in [t for t in TOKENIZERS if t in set(c.tokenizer)]:
    v = c[c.tokenizer == t].set_index("lang").reindex(langs).pct_context_lost_128k
    ax.plot(range(len(langs)), v, marker="o", label=ps.TOKENIZER_LABELS[t])
ax.set_xticks(range(len(langs)), [ps.lang_label(l) for l in langs], rotation=35, ha="right")
ax.set_ylabel("% of context window lost vs English")
ax.set_ylim(0, 100)
ax.set_title("Same content, less room: effective context lost to the tax")
ax.legend(frameon=False, fontsize=8, ncol=2)
ps.save(fig, "fig_context_loss")
