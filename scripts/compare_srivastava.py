"""D2 sanity check: our cl100k / o200k taxes vs Srivastava (2026, arXiv 2607.24276), Table 2.

Their tax = word-fertility ratio vs English on FLORES-200 dev (997 sents). Ours (tax_vs_eng) =
total-token ratio on devtest (1,012 sents). We compute both so the gap can be attributed.
Their single-byte % (cl100k) is also compared with two readings of "single byte":
  ours  = token bytes are not a complete UTF-8 char (the plan's definition)
  1byte = token is exactly one byte long (includes ASCII punctuation; may be theirs, given English = 9.3%)
"""
import pandas as pd

from tokentax.metrics import summarize_e1
from tokentax.tokenizers import get_tokenizer

PAPER = pd.DataFrame({
    "lang": ["urd_Arab", "hin_Deva", "mar_Deva", "ben_Beng", "pan_Guru", "guj_Gujr", "tam_Taml",
             "tel_Telu", "kan_Knda", "mal_Mlym", "eng_Latn"],
    "paper_cl100k": [3.45, 4.08, 5.84, 6.52, 6.69, 8.00, 9.89, 10.68, 12.06, 13.04, 1.0],
    "paper_o200k": [1.31, 1.34, 2.11, 1.91, 2.23, 1.87, 2.57, 2.49, 2.72, 2.85, 1.0],
    "paper_cl100k_single_byte_pct": [13.3, 27.5, 29.2, 37.2, 39.1, 39.5, 33.9, 40.9, 41.0, 42.7, 9.3],
})

e1 = pd.read_csv("results/e1_intrinsic.csv")
s = summarize_e1(e1)
eng_fert = s[s.lang == "eng_Latn"].set_index("tokenizer").fertility
s["fert_ratio"] = s.fertility / s.tokenizer.map(eng_fert)

out = PAPER.copy()
for t in ["cl100k", "o200k"]:
    st = s[s.tokenizer == t].set_index("lang")
    out[f"ours_{t}_token_ratio"] = out.lang.map(st.tax_vs_eng).round(2)
    out[f"ours_{t}_fert_ratio"] = out.lang.map(st.fertility / eng_fert[t]).round(2)
out["ours_cl100k_single_byte_pct"] = out.lang.map(
    s[s.tokenizer == "cl100k"].set_index("lang").pct_single_byte).round(1)

# Alternative reading: share of tokens that are exactly 1 byte long, on the same devtest sentences.
from tokentax.data import load_flores

tok = get_tokenizer("cl100k")
one = {}
for l in out.lang:
    ids = [i for x in load_flores(l) for i in tok.encode(x)]
    one[l] = 100 * sum(len(tok.token_bytes(i)) == 1 for i in ids) / len(ids)
out["ours_cl100k_1byte_len_pct"] = out.lang.map(one).round(1)

out["gap_fert_vs_paper_cl100k_pct"] = (100 * (out.ours_cl100k_fert_ratio / out.paper_cl100k - 1)).round(1)
out.to_csv("results/srivastava_comparison.csv", index=False)
print(out.to_string(index=False))
