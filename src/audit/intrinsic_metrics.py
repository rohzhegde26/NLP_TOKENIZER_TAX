"""Intrinsic Tokenizer Tax Engine: Fertility, Parity, Gini, Bytes/Token, Unmerged Byte Rate."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class BaseTokenizerWrapper:
    """Unified wrapper around tiktoken and Hugging Face tokenizers."""

    def __init__(self, name: str, identifier: str, tok_type: str = "huggingface"):
        self.name = name
        self.identifier = identifier
        self.tok_type = tok_type
        self._tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        if self.tok_type == "tiktoken":
            import tiktoken
            self._tokenizer = tiktoken.get_encoding(self.identifier)
        else:
            from transformers import AutoTokenizer
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.identifier, trust_remote_code=True
                )
            except Exception as e:
                # Fallback for Gemma or IndicBERT
                if "gemma" in self.identifier.lower():
                    self._tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-2b")
                elif "indicbert" in self.identifier.lower():
                    self._tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
                else:
                    raise e

    def encode(self, text: str) -> List[int]:
        if self.tok_type == "tiktoken":
            return self._tokenizer.encode(text)
        else:
            return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        if self.tok_type == "tiktoken":
            return self._tokenizer.decode(token_ids)
        else:
            return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def is_single_byte_token(self, token_id: int) -> bool:
        """Determine if a token represents an unmerged single-byte fallback."""
        if self.tok_type == "tiktoken":
            try:
                b = self._tokenizer.decode_single_token_bytes(token_id)
                return len(b) == 1
            except Exception:
                return False
        else:
            try:
                tok_str = self._tokenizer.convert_ids_to_tokens(token_id)
                if isinstance(tok_str, bytes):
                    return len(tok_str) == 1
                if tok_str.startswith("<0x") and tok_str.endswith(">") and len(tok_str) == 6:
                    return True
                if tok_str.startswith("byte:") and len(tok_str) <= 7:
                    return True
                # Check if decoded string is a single byte non-printable/fallback
                raw = tok_str.encode("utf-8", errors="ignore")
                return len(raw) == 1 and not (32 <= raw[0] <= 126)
            except Exception:
                return False

    @property
    def vocab_size(self) -> int:
        if self.tok_type == "tiktoken":
            return self._tokenizer.n_vocab
        return len(self._tokenizer)


def compute_sentence_metrics(
    text: str,
    tokenizer: BaseTokenizerWrapper
) -> Dict[str, Any]:
    """Compute per-sentence metrics: tokens, words, bytes, single bytes."""
    words = text.split()
    word_count = max(1, len(words))
    utf8_bytes = len(text.encode("utf-8"))
    char_count = len(text)

    token_ids = tokenizer.encode(text)
    token_count = len(token_ids)
    single_byte_count = sum(1 for tid in token_ids if tokenizer.is_single_byte_token(tid))

    return {
        "word_count": word_count,
        "token_count": token_count,
        "utf8_bytes": utf8_bytes,
        "char_count": char_count,
        "single_byte_count": single_byte_count,
    }


def compute_gini(values: List[float]) -> float:
    """Compute Gini inequality coefficient across language values:
    
    Gini = sum_{i=1}^M sum_{j=1}^M |P_i - P_j| / (2 * M * sum_{i=1}^M P_i)
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    if n == 0 or np.sum(arr) == 0:
        return 0.0
    abs_diff = np.abs(np.subtract.outer(arr, arr))
    return float(np.sum(abs_diff) / (2.0 * n * np.sum(arr)))


def audit_corpus_metrics(
    lang_sentences: Dict[str, List[str]],
    tokenizer: BaseTokenizerWrapper,
    baseline_lang: str = "eng_Latn"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute full intrinsic audit across all languages for a tokenizer on parallel sentences."""
    if baseline_lang not in lang_sentences:
        raise ValueError(f"Baseline language '{baseline_lang}' must be present in parallel corpus.")

    eng_sentences = lang_sentences[baseline_lang]
    n_sentences = len(eng_sentences)

    # Pre-tokenize baseline English sentences
    eng_token_counts = [
        max(1, len(tokenizer.encode(s))) for s in eng_sentences
    ]

    records = []
    lang_parity_map = {}

    for lang, sents in lang_sentences.items():
        total_tokens = 0
        total_words = 0
        total_bytes = 0
        total_chars = 0
        total_single_bytes = 0
        parity_ratios = []

        for i, s in enumerate(sents):
            metrics = compute_sentence_metrics(s, tokenizer)
            t_cnt = metrics["token_count"]
            total_tokens += t_cnt
            total_words += metrics["word_count"]
            total_bytes += metrics["utf8_bytes"]
            total_chars += metrics["char_count"]
            total_single_bytes += metrics["single_byte_count"]

            if i < len(eng_token_counts):
                parity_ratios.append(t_cnt / eng_token_counts[i])

        fertility = total_tokens / max(1, total_words)
        mean_parity = float(np.mean(parity_ratios)) if parity_ratios else 1.0
        bytes_per_token = total_bytes / max(1, total_tokens)
        chars_per_token = total_chars / max(1, total_tokens)
        single_byte_rate = (total_single_bytes / max(1, total_tokens)) * 100.0

        lang_parity_map[lang] = mean_parity

        records.append({
            "tokenizer": tokenizer.name,
            "language": lang,
            "fertility": round(fertility, 3),
            "parity_ratio": round(mean_parity, 3),
            "bytes_per_token": round(bytes_per_token, 3),
            "chars_per_token": round(chars_per_token, 3),
            "single_byte_rate_pct": round(single_byte_rate, 2),
            "total_tokens": total_tokens,
            "total_words": total_words,
            "total_bytes": total_bytes,
        })

    df = pd.DataFrame(records)

    # Compute Gini across Indic languages (+ English)
    all_parities = list(lang_parity_map.values())
    indic_parities = [p for l, p in lang_parity_map.items() if l != baseline_lang]

    gini_all = compute_gini(all_parities)
    gini_indic = compute_gini(indic_parities)
    mean_indic_parity = float(np.mean(indic_parities)) if indic_parities else 1.0

    summary = {
        "tokenizer": tokenizer.name,
        "mean_indic_parity": round(mean_indic_parity, 3),
        "gini_all": round(gini_all, 4),
        "gini_indic": round(gini_indic, 4),
        "mean_bytes_per_token": round(float(df["bytes_per_token"].mean()), 3),
        "mean_single_byte_rate_pct": round(float(df["single_byte_rate_pct"].mean()), 2),
    }

    return df, summary
