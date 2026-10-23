"""Inference-time BPE-dropout (owner: Rohith). STUB with the Day-1 signature."""
from __future__ import annotations


def resegment(tokenizer, ids: list[int], p: float, seed: int) -> list[int]:
    """Finer split of `ids` that decodes to the same string. Stub: identity (valid at p=0)."""
    if p == 0:
        return list(ids)
    raise NotImplementedError("Rohith: implement in lane-c")
