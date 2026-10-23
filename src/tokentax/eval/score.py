"""Option scoring (owner: Rohit). STUB: signatures fixed by the Day-1 contract."""
from __future__ import annotations


def load_model(name: str):
    """Load a model (4-bit) + its tokenizer. Returns an opaque model handle."""
    raise NotImplementedError("Rohit: implement in lane-b")


def score(model, passage_ids: list[int], question: str, options: list[str]) -> int:
    """Predicted option 1-4 by next-token probability of ' A'/' B'/' C'/' D'.

    Takes passage token ids (not text) so E3 truncation and E4 re-segmentation plug in.
    """
    raise NotImplementedError("Rohit: implement in lane-b")
