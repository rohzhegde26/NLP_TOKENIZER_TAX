"""Gap decomposition: stacked bars of d_ctx, d_frag, d_res per language, one panel per model.

Reads results/decomposition.csv (PROPOSED schema, not yet agreed with lane C):
  model,lang,d_ctx,d_ctx_lo,d_ctx_hi,d_frag,d_frag_lo,d_frag_hi,d_res,d_res_lo,d_res_hi,share_capped
Each d_* is an accuracy difference (0-1) from Algorithm 2 in PLANNING.md; *_lo/_hi are 95% bootstrap CIs.
"""
import matplotlib.pyplot as plt
import numpy as np

from tokentax import plotstyle as ps
from tokentax.schemas import load

TERMS = [("d_ctx", "Context lost (Δctx)", "#e9a23b"),
         ("d_frag", "Fragmentation (Δfrag)", "#c8553d"),
         ("d_res", "Competence / data (Δres)", "#2a6f97")]

ps.apply()
d = load("decomposition")
models = list(dict.fromkeys(d.model))
fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 3.8), sharey=True, squeeze=False)
for ax, m in zip(axes[0], models):
    dm = d[d.model == m].set_index("lang")
    langs = [l for l in ps.LANG_COLORS if l in dm.index and l != "eng_Latn"]
    x = np.arange(len(langs))
    pos, neg = np.zeros(len(langs)), np.zeros(len(langs))
    for col, label, color in TERMS:
        v = dm.loc[langs, col].to_numpy()
        base = np.where(v >= 0, pos, neg)  # stack negatives below zero
        ax.bar(x, v, 0.6, bottom=base, color=color, label=label)
        pos += np.clip(v, 0, None)
        neg += np.clip(v, None, 0)
    total = dm.loc[langs, ["d_ctx", "d_frag", "d_res"]].sum(axis=1)
    ax.scatter(x, total, marker="_", s=300, color="black", zorder=3, label="Total gap Δ")
    for i, l in enumerate(langs):
        if dm.loc[l, "share_capped"] > 0.05:
            ax.text(i, pos[i] + 0.01, f"{100 * dm.loc[l, 'share_capped']:.0f}% capped", ha="center", fontsize=7)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x, [ps.lang_label(l) for l in langs], rotation=35, ha="right")
    ax.set_title(m)
axes[0][0].set_ylabel("Accuracy gap vs English")
axes[0][0].legend(frameon=False, fontsize=8)
ps.save(fig, "fig_decomposition")
