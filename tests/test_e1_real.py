"""Checks on the real E1 output (skipped until results/e1_intrinsic.csv exists)."""
import os

import pandas as pd
import pytest

from tokentax.metrics import summarize_e1
from tokentax.schemas import check

pytestmark = pytest.mark.skipif(not os.path.exists("results/e1_intrinsic.csv"), reason="E1 not run yet")


def test_e1_schema_and_english_tax():
    df = pd.read_csv("results/e1_intrinsic.csv")
    check(df, "e1_intrinsic")
    s = summarize_e1(df)
    assert (s[s.lang == "eng_Latn"].tax_vs_eng == 1.0).all()
    assert (df.groupby(["tokenizer", "lang"]).size() == 1012).all()
    assert (df.n_tokens >= 1).all()
