"""Dataset loaders. FLORES part: Saket. Belebele part: Rohit."""
from __future__ import annotations

import unicodedata

import pandas as pd

FLORES_REPO = "openlanguagedata/flores_plus"  # gated: accept terms on the Hub
BELEBELE_REPO = "facebook/belebele"


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def load_flores(lang: str) -> list[str]:
    """The 1,012 FLORES-200 devtest sentences for `lang` (FLORES code), NFC-normalized, in id order."""
    from datasets import load_dataset

    ds = load_dataset(FLORES_REPO, lang, split="devtest")
    if "id" in ds.column_names:
        ds = ds.sort("id")
    sents = [nfc(t) for t in ds["text"]]
    assert len(sents) == 1012, f"{lang}: expected 1012 devtest sentences, got {len(sents)}"
    return sents


# --- Belebele (owner: Rohit) -------------------------------------------------
def load_belebele(lang: str, n: int = 300) -> pd.DataFrame:
    """First `n` Belebele questions in a fixed cross-language order.

    Columns: qid, passage, question, opt1..opt4, gold (1-4).
    STUB by Saket so others can import; Rohit owns and may refine (keep the columns).
    """
    from datasets import load_dataset

    df = load_dataset(BELEBELE_REPO, lang, split="test").to_pandas()
    df = df.sort_values(["link", "question_number"], kind="stable").reset_index(drop=True)
    df["qid"] = df["link"].astype(str) + "#" + df["question_number"].astype(str)
    out = pd.DataFrame({
        "qid": df["qid"],
        "passage": df["flores_passage"].map(nfc),
        "question": df["question"].map(nfc),
        **{f"opt{i}": df[f"mc_answer{i}"].map(nfc) for i in range(1, 5)},
        "gold": df["correct_answer_num"].astype(int),
    })
    return out.head(n)
