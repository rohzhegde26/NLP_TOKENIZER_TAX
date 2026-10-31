"""Context Erosion and Monetary Tokenizer Tax Analysis."""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_context_erosion(
    parity_df: pd.DataFrame,
    windows: List[int] = [512, 2048, 4096],
    avg_eng_tokens_per_sentence: float = 25.0
) -> pd.DataFrame:
    """Calculate effective context percentage and sentence capacity under fixed token windows.
    
    Effective Context % = 100 / P_L
    Sentence capacity W_L = floor(W / (avg_eng_tokens_per_sentence * P_L))
    """
    records = []
    for _, row in parity_df.iterrows():
        tokenizer = row["tokenizer"]
        lang = row["language"]
        p_l = max(0.01, float(row["parity_ratio"]))
        effective_context_pct = round(100.0 / p_l, 2)

        record = {
            "tokenizer": tokenizer,
            "language": lang,
            "parity_ratio": p_l,
            "effective_context_pct": effective_context_pct,
        }

        for w in windows:
            capacity = int(w / (avg_eng_tokens_per_sentence * p_l))
            record[f"capacity_sentences_w{w}"] = capacity

        records.append(record)

    return pd.DataFrame(records)


def compute_monetary_tax(
    parity_df: pd.DataFrame,
    tokenizers_config: Dict[str, Any],
    eng_tokens_per_1k_words: float = 1350.0
) -> pd.DataFrame:
    """Calculate the dollar cost multiplier and API bill per 1,000 semantic words.
    
    Cost per 1k semantic words = (eng_tokens_per_1k_words * P_L) * (rate_per_1k / 1000)
    """
    records = []
    for _, row in parity_df.iterrows():
        tok_name = row["tokenizer"]
        lang = row["language"]
        p_l = max(0.01, float(row["parity_ratio"]))

        # Match tokenizer pricing rate
        rate_per_1k = 0.0025  # default $2.50 per 1M tokens
        for k, v in tokenizers_config.items():
            if v["name"] == tok_name or k in tok_name.lower() or v.get("identifier") in tok_name:
                rate_per_1k = float(v.get("pricing_per_1k_tokens_usd", 0.0025))
                break

        tokens_per_1k_semantic_words = eng_tokens_per_1k_words * p_l
        cost_usd = (tokens_per_1k_semantic_words / 1000.0) * rate_per_1k
        cost_multiplier = p_l

        records.append({
            "tokenizer": tok_name,
            "language": lang,
            "parity_ratio": p_l,
            "rate_per_1k_usd": rate_per_1k,
            "tokens_per_1k_words": round(tokens_per_1k_semantic_words, 1),
            "dollar_cost_per_1k_words": round(cost_usd, 5),
            "monetary_cost_multiplier": round(cost_multiplier, 3),
        })

    return pd.DataFrame(records)
