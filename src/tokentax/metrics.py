"""Intrinsic metrics for E1 (owner: Saket)."""
from __future__ import annotations

import pandas as pd

from .langs import ENGLISH


def sentence_row(tok, text: str) -> dict:
    ids = tok.encode(text)
    return {
        "n_tokens": len(ids),
        "n_words": len(text.split()),
        "n_chars": len(text),
        "n_bytes": len(text.encode("utf-8")),
        "n_single_byte": sum(tok.is_single_byte(i) for i in ids),
    }


def summarize_e1(e1: pd.DataFrame) -> pd.DataFrame:
    """Per (tokenizer, lang): totals, tax vs English (same sentences), fertility, bytes/token."""
    g = e1.groupby(["tokenizer", "lang"], as_index=False)[
        ["n_tokens", "n_words", "n_chars", "n_bytes", "n_single_byte"]].sum()
    eng = g[g.lang == ENGLISH].set_index("tokenizer")
    g["eng_tokens"] = g.tokenizer.map(eng.n_tokens)
    g["eng_words"] = g.tokenizer.map(eng.n_words)
    g["tax_vs_eng"] = g.n_tokens / g.eng_tokens
    g["fertility"] = g.n_tokens / g.n_words
    g["bytes_per_token"] = g.n_bytes / g.n_tokens
    g["pct_single_byte"] = 100 * g.n_single_byte / g.n_tokens
    return g
