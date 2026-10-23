"""Counterfactual tax decomposition (Lane C, D5).

Estimates the three terms of the causal gap decomposition:
- delta_ctx: context lost due to token truncation
- delta_frag: fragmentation penalty (matched fragmentation of English)
- delta_res: residual competence and training data gap
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tokentax.eval.resegment import match_fragment
from tokentax.eval.score import score
from tokentax.stats import bootstrap_ci


def decompose_items(
    model_scorer,
    tokenizer,
    df_en: pd.DataFrame,
    df_lang: pd.DataFrame,
    rng: np.random.Generator | int = 0,
    n_boot: int = 1000,
) -> dict:
    """Run counterfactual decomposition across paired Belebele items."""
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    # Align on qid
    merged = pd.merge(df_en, df_lang, on="qid", suffixes=("_en", "_l"))
    rows = []

    for _, row in merged.iterrows():
        en_passage = str(row["passage_en"])
        l_passage = str(row["passage_l"])
        en_ids = tokenizer.encode(en_passage)
        l_ids = tokenizer.encode(l_passage)

        options_en = [str(row[f"opt{i}_en"]) for i in range(1, 5)]
        options_l = [str(row[f"opt{i}_l"]) for i in range(1, 5)]
        gold_en = int(row["gold_en"])
        gold_l = int(row["gold_l"])

        # 1. A(en, canonical, full)
        pred_en = score(model_scorer, en_ids, str(row["question_en"]), options_en)
        a_en = float(pred_en == gold_en)

        # 2. A(l, canonical, full)
        pred_l_full = score(model_scorer, l_ids, str(row["question_l"]), options_l)
        a_l_full = float(pred_l_full == gold_l)

        # 3. A(l, canonical, budget B = |en|)
        budget = len(en_ids)
        l_cut = l_ids[:budget]
        pred_l_cut = score(model_scorer, l_cut, str(row["question_l"]), options_l)
        a_l_cut = float(pred_l_cut == gold_l)

        # 4. A(en, s^l, full) - matched fragmentation of English
        en_frag, capped = match_fragment(tokenizer, en_ids, len(l_ids), rng=rng)
        pred_en_frag = score(model_scorer, en_frag, str(row["question_en"]), options_en)
        a_en_frag = float(pred_en_frag == gold_en)

        rows.append({
            "a_en": a_en,
            "a_l_full": a_l_full,
            "a_l_cut": a_l_cut,
            "a_en_frag": a_en_frag,
            "diff_ctx": a_l_full - a_l_cut,
            "diff_frag": a_en - a_en_frag,
            "diff_res": a_en_frag - a_l_full,
            "capped": float(capped),
        })

    rdf = pd.DataFrame(rows)
    d_ctx = float(rdf["diff_ctx"].mean())
    d_frag = float(rdf["diff_frag"].mean())
    d_res = float(rdf["diff_res"].mean())

    ctx_low, ctx_high = bootstrap_ci(rdf["diff_ctx"], n_boot=n_boot, seed=42)
    frag_low, frag_high = bootstrap_ci(rdf["diff_frag"], n_boot=n_boot, seed=42)
    res_low, res_high = bootstrap_ci(rdf["diff_res"], n_boot=n_boot, seed=42)

    return {
        "d_ctx": d_ctx,
        "d_ctx_ci": (ctx_low, ctx_high),
        "d_frag": d_frag,
        "d_frag_ci": (frag_low, frag_high),
        "d_res": d_res,
        "d_res_ci": (res_low, res_high),
        "share_capped": float(rdf["capped"].mean()),
        "n_items": len(rdf),
    }
