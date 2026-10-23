"""Tax heatmap: language x tokenizer, token ratio vs English."""
import matplotlib.pyplot as plt

from tokentax import plotstyle as ps
from tokentax.metrics import summarize_e1
from tokentax.schemas import load
from tokentax.tokenizers import TOKENIZERS

ps.apply()
s = summarize_e1(load("e1_intrinsic"))
toks = [t for t in TOKENIZERS if t in set(s.tokenizer)]
m = s.pivot(index="lang", columns="tokenizer", values="tax_vs_eng").loc[ps.ordered(s.lang), toks]
fig, ax = plt.subplots(figsize=(1.0 * len(toks) + 2, 0.42 * len(m) + 1))
im = ax.imshow(m.values, cmap="OrRd", aspect="auto")
ax.set_xticks(range(len(toks)), [ps.TOKENIZER_LABELS[t] for t in toks], rotation=35, ha="right")
ax.set_yticks(range(len(m)), [ps.lang_label(l) for l in m.index])
vmax = m.values.max()
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        v = m.values[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white" if v > vmax * 0.6 else "black")
fig.colorbar(im, ax=ax, label="Tokens relative to English")
ax.set_title("Tokenizer tax on FLORES-200 devtest")
ps.save(fig, "fig_tax_heatmap")
