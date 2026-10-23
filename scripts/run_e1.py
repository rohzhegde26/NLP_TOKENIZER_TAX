"""E1: 7 tokenizers x 13 languages on FLORES-200 devtest -> results/e1_intrinsic.csv (+ e1_summary.csv)."""
import argparse
import sys

import pandas as pd

from tokentax.data import load_flores
from tokentax.langs import AUDIT_LANGS
from tokentax.metrics import sentence_row, summarize_e1
from tokentax.schemas import COLUMNS, check
from tokentax.tokenizers import TOKENIZERS, get_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizers", nargs="*", default=TOKENIZERS)
ap.add_argument("--langs", nargs="*", default=AUDIT_LANGS)
a = ap.parse_args()

texts = {l: load_flores(l) for l in a.langs}
rows, skipped = [], []
for t in a.tokenizers:
    try:
        tok = get_tokenizer(t)
    except Exception as e:  # gated licence not approved yet, etc.
        print(f"SKIP {t}: {type(e).__name__}: {str(e)[:150]}", file=sys.stderr)
        skipped.append(t)
        continue
    for l in a.langs:
        for i, s in enumerate(texts[l]):
            rows.append({"tokenizer": t, "lang": l, "sent_id": i, **sentence_row(tok, s)})
    print(f"done {t}")

df = pd.DataFrame(rows)[COLUMNS["e1_intrinsic"]]
check(df, "e1_intrinsic")
df.to_csv("results/e1_intrinsic.csv", index=False)
summ = summarize_e1(df)
summ.to_csv("results/e1_summary.csv", index=False)
print(summ.pivot(index="lang", columns="tokenizer", values="tax_vs_eng").round(2).to_string())
if skipped:
    print("skipped (rerun when available):", skipped)
