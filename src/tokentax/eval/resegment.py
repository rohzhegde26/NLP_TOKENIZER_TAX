"""Inference-time BPE-dropout and fragmentation intervention (owner: Rohith).

Lane C core contracts:
- resegment(tokenizer, ids, p, seed): randomly splits tokens with probability p into valid vocab pieces
- match_fragment(tokenizer, en_ids, target_len, rng): Algorithm 1, matched fragmentation of English
- to_single_bytes(tokenizer, ids): splits multi-byte tokens into single-byte UTF-8 tokens (mechanism check)
"""
from __future__ import annotations

import numpy as np


def split_token(tokenizer, tid: int, rng: np.random.Generator) -> list[int]:
    """Split a single token into two or more finer tokens in the vocabulary."""
    b = tokenizer.token_bytes(tid)
    if len(b) <= 1:
        return [tid]

    # Find valid cuts where both pieces are in the vocabulary
    cuts = [k for k in range(1, len(b)) if tokenizer.has(b[:k]) and tokenizer.has(b[k:])]
    if cuts:
        k = int(rng.choice(cuts))
        return [tokenizer.id(b[:k]), tokenizer.id(b[k:])]

    # Fallback for byte-level BPE: split into single bytes if available
    try:
        single_bytes = [tokenizer.id(bytes([x])) for x in b]
        return single_bytes
    except (KeyError, Exception):
        return [tid]


def resegment(tokenizer, ids: list[int], p: float, seed: int = 0) -> list[int]:
    """Finer split of `ids` that decodes to the identical string.

    For each token in `ids`, splits it into subword/byte pieces with probability `p`.
    At p=0, returns the exact original list(ids).
    """
    if p == 0 or not ids:
        return list(ids)

    rng = np.random.default_rng(seed)
    out: list[int] = []
    for tid in ids:
        if rng.random() < p:
            out.extend(split_token(tokenizer, tid, rng))
        else:
            out.append(tid)

    return out


def to_single_bytes(tokenizer, ids: list[int]) -> list[int]:
    """Split every multi-byte token in `ids` into raw single-byte tokens.

    Tests the failed-merge mechanism (E4a single-byte condition).
    """
    out: list[int] = []
    for tid in ids:
        b = tokenizer.token_bytes(tid)
        if len(b) <= 1:
            out.append(tid)
        else:
            try:
                out.extend([tokenizer.id(bytes([x])) for x in b])
            except KeyError:
                out.append(tid)
    return out


def match_fragment(
    tokenizer, en_ids: list[int], target_len: int, rng: np.random.Generator | int = 0
) -> tuple[list[int], bool]:
    """Algorithm 1: match_fragment.

    Makes English `en_ids` as fragmented as a target language (length target_len).
    Returns (resegmented_ids, capped).
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    s = list(en_ids)
    while len(s) < target_len:
        splittable = [i for i, t in enumerate(s) if len(tokenizer.token_bytes(t)) > 1]
        if not splittable:
            # Already character/byte level: cap reached
            return s, True

        i = int(rng.choice(splittable))
        b = tokenizer.token_bytes(s[i])
        cuts = [k for k in range(1, len(b)) if tokenizer.has(b[:k]) and tokenizer.has(b[k:])]
        if cuts:
            k = int(rng.choice(cuts))
            s[i : i + 1] = [tokenizer.id(b[:k]), tokenizer.id(b[k:])]
        else:
            try:
                single_bytes = [tokenizer.id(bytes([x])) for x in b]
                s[i : i + 1] = single_bytes
            except KeyError:
                break

    return s, False
