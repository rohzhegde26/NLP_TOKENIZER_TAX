"""E3 grouped bars: accuracy for full vs eqtok_1.0 vs eqtok_0.5 per language, one panel per model, 95% CIs."""
import matplotlib.pyplot as plt
import numpy as np

from tokentax import plotstyle as ps
from tokentax.schemas import CONDITIONS, load
from tokentax.stats import bootstrap_ci

ps.apply()
d = load("e3_results")
models = list(dict.fromkeys(d.model))
langs = [l for l in ps.LANG_COLORS if l in set(d.lang)]
fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 3.8), sharey=True, squeeze=False)
w = 0.27
for ax, m in zip(axes[0], models):
    for k, c in enumerate(CONDITIONS):
        acc, lo, hi = [], [], []
        for l in langs:
            x = d[(d.model == m) & (d.lang == l) & (d.condition == c)].correct.astype(float)
            if len(x):
                a, ci = x.mean(), bootstrap_ci(x)
            else:
                a, ci = np.nan, (np.nan, np.nan)
            acc.append(a)
            lo.append(a - ci[0])
            hi.append(ci[1] - a)
        ax.bar(np.arange(len(langs)) + (k - 1) * w, acc, w, yerr=[lo, hi], capsize=2,
               color=ps.CONDITION_COLORS[c], label=c)
    ax.axhline(0.25, ls=":", c="grey", lw=1)
    ax.set_xticks(range(len(langs)), [ps.lang_label(l) for l in langs], rotation=35, ha="right")
    ax.set_title(m)
axes[0][0].set_ylabel("Belebele accuracy")
axes[0][0].legend(frameon=False, fontsize=8)
ps.save(fig, "fig_e3")
