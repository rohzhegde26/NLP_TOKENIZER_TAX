"""Write all four result CSVs with random values in the contract shape (results/fake/ by default)."""
import argparse
import os

import numpy as np
import pandas as pd

from tokentax.langs import AUDIT_LANGS, GPU_LANGS
from tokentax.schemas import COLUMNS, CONDITIONS, E4_LANGS, E4_RATES, MODELS, check
from tokentax.tokenizers import TOKENIZERS

rng = np.random.default_rng(0)
TAX = {l: 1.0 + 0.4 * i for i, l in enumerate(AUDIT_LANGS)}  # monotone fake tax, English = 1


def e1():
    rows = []
    for t in TOKENIZERS:
        for l in AUDIT_LANGS:
            for s in range(1012):
                w = int(rng.integers(10, 35))
                rows.append([t, l, s, int(w * 1.3 * TAX[l]), w, w * 6, w * 6 * (1 if l == "eng_Latn" else 3),
                             0 if l == "eng_Latn" else int(rng.integers(0, w))])
    return pd.DataFrame(rows, columns=COLUMNS["e1_intrinsic"])


def e2():
    rows = []
    for t in TOKENIZERS:
        for l in AUDIT_LANGS:
            tax = TAX[l] * rng.uniform(0.8, 1.2) if l != "eng_Latn" else 1.0
            usd = 0.0033 * tax
            rows.append([t, l, round(tax, 3), usd, usd * 84.0, round(100 * (1 - 1 / tax), 2), "2026-10-25"])
    return pd.DataFrame(rows, columns=COLUMNS["e2_cost"])


def e3():
    rows = []
    for m in MODELS:
        for l in GPU_LANGS:
            for c in CONDITIONS:
                for q in range(300):
                    full = int(rng.integers(100, 200) * TAX[l])
                    budget = full // int(TAX[l]) if c != "full" else full
                    budget = budget // 2 if c == "eqtok_0.5" else budget
                    gold = int(rng.integers(1, 5))
                    pred = gold if rng.random() < 0.8 - 0.05 * GPU_LANGS.index(l) else int(rng.integers(1, 5))
                    rows.append([m, l, c, budget, f"q{q}", full, min(full, budget), bool(rng.random() < 0.8),
                                 pred, gold, pred == gold])
    return pd.DataFrame(rows, columns=COLUMNS["e3_results"])


def e4():
    rows = []
    for l in E4_LANGS:
        for p in E4_RATES:
            for q in range(300):
                can = int(rng.integers(100, 300))
                gold = int(rng.integers(1, 5))
                pred = gold if rng.random() < 0.8 - p * 0.4 else int(rng.integers(1, 5))
                rows.append(["qwen2.5-3b", l, p, 0, f"q{q}", can, int(can * (1 + p)), pred, gold, pred == gold])
    return pd.DataFrame(rows, columns=COLUMNS["e4_results"])


def decomposition():
    """PROPOSED schema (not a Day-1 contract): output of Algorithm 2 per (model, lang)."""
    rows = []
    for m in MODELS:
        for l in GPU_LANGS[1:]:
            r = {"model": m, "lang": l}
            for k, mu in [("d_ctx", 0.08), ("d_frag", 0.05), ("d_res", 0.12)]:
                v = mu * rng.uniform(0.5, 1.5)
                r.update({k: v, f"{k}_lo": v - 0.03, f"{k}_hi": v + 0.03})
            r["share_capped"] = rng.uniform(0, 0.1)
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/fake")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, fn in [("e1_intrinsic", e1), ("e2_cost", e2), ("e3_results", e3), ("e4_results", e4)]:
        df = fn()
        check(df, name)
        df.to_csv(f"{a.out}/{name}.csv", index=False)
        print(f"wrote {a.out}/{name}.csv ({len(df)} rows)")
    decomposition().to_csv(f"{a.out}/decomposition.csv", index=False)
    print(f"wrote {a.out}/decomposition.csv (proposed schema)")
