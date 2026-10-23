"""E2: cost per 1k English-equivalent words and context loss -> results/e2_cost.csv, e2_context.csv."""
import argparse
import glob

import pandas as pd

from tokentax.cost import e2_table, load_prices
from tokentax.metrics import summarize_e1
from tokentax.schemas import check

ap = argparse.ArgumentParser()
ap.add_argument("--prices", default=sorted(glob.glob("configs/prices_*.csv"))[-1])
a = ap.parse_args()

summary = summarize_e1(pd.read_csv("results/e1_intrinsic.csv"))
cost, ctx = e2_table(summary, load_prices(a.prices))
check(cost, "e2_cost")
cost.to_csv("results/e2_cost.csv", index=False)
ctx.to_csv("results/e2_context.csv", index=False)
print(f"prices: {a.prices}")
print(cost.pivot(index="lang", columns="tokenizer", values="inr_per_1k_eng_words").round(3).to_string())
