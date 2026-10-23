"""Cost and context-window maths for E2 (owner: Saket)."""
from __future__ import annotations

import pandas as pd

WINDOWS = [8_192, 128_000]


def load_prices(path: str) -> pd.DataFrame:
    """Price sheet: tokenizer, model, usd_per_m_input, source_url, snapshot_date, usd_inr."""
    return pd.read_csv(path, comment="#")


def e2_table(summary: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (e2_cost in contract columns, e2_context with words-that-fit per window).

    English-equivalent words: the English FLORES word count of the same content.
    tokens per 1k English-eq words = 1000 * lang_tokens / eng_words.
    """
    s = summary.copy()
    s["tok_per_1k_eng_words"] = 1000 * s.n_tokens / s.eng_words
    # One reference price per tokenizer (first row in the sheet); tokenizers with no API get NaN.
    p = prices.drop_duplicates("tokenizer").set_index("tokenizer")
    s["usd_per_1k_eng_words"] = s.tok_per_1k_eng_words * s.tokenizer.map(p.usd_per_m_input) / 1e6
    s["inr_per_1k_eng_words"] = s.usd_per_1k_eng_words * s.tokenizer.map(p.usd_inr)
    s["pct_context_lost_128k"] = 100 * (1 - 1 / s.tax_vs_eng)
    s["price_snapshot_date"] = s.tokenizer.map(p.snapshot_date)
    cost = s[["tokenizer", "lang", "tax_vs_eng", "usd_per_1k_eng_words", "inr_per_1k_eng_words",
              "pct_context_lost_128k", "price_snapshot_date"]]
    ctx = s[["tokenizer", "lang", "tax_vs_eng"]].copy()
    for w in WINDOWS:
        ctx[f"eng_words_fit_{w}"] = w * s.eng_words / s.n_tokens
    ctx["pct_window_lost"] = 100 * (1 - 1 / s.tax_vs_eng)
    return cost, ctx
