"""E4: accuracy vs induced fertility ratio (n_tokens_used / n_tokens_canonical), one line per language."""
import matplotlib.pyplot as plt
import numpy as np

from tokentax import plotstyle as ps
from tokentax.schemas import load
from tokentax.stats import bootstrap_ci

ps.apply()
d = load("e4_results")
d["ratio"] = d.n_tokens_used / d.n_tokens_canonical
models = list(dict.fromkeys(d.model))
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.8), sharey=True, squeeze=False)
for ax, m in zip(axes[0], models):
    dm = d[d.model == m]
    for l in [l for l in ps.LANG_COLORS if l in set(dm.lang)]:
        g = dm[dm.lang == l].groupby("p")
        x, acc = g.ratio.mean(), g.correct.mean()
        ci = np.array([bootstrap_ci(v.astype(float)) for _, v in g.correct])
        ax.errorbar(x, acc, yerr=[acc - ci[:, 0], ci[:, 1] - acc], marker="o", capsize=2,
                    color=ps.LANG_COLORS[l], label=ps.lang_label(l))
    ax.axhline(0.25, ls=":", c="grey", lw=1)
    ax.set_xlabel("Induced fertility (tokens / canonical tokens)")
    ax.set_title(m)
axes[0][0].set_ylabel("Belebele accuracy")
axes[0][0].legend(frameon=False, fontsize=8)
ps.save(fig, "fig_e4")
