"""Analyze E4 results: compute slopes, bootstrap CIs, and language comparisons (Lane C, D6).

Reads results/e4_results.csv (or falls back to results/fake/e4_results.csv)
and outputs results/e4_analysis.csv and a markdown summary.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from tokentax.schemas import load
from tokentax.stats import bootstrap_ci, bootstrap_slope_ci, fit_slope


def analyze_e4(in_path: str | None = None, out_path: str = "results/e4_analysis.csv") -> pd.DataFrame:
    if in_path and os.path.exists(in_path):
        df = pd.read_csv(in_path)
    else:
        df = load("e4_results")

    df["ratio"] = df["n_tokens_used"] / df["n_tokens_canonical"]

    records = []
    models = list(dict.fromkeys(df["model"]))
    langs = list(dict.fromkeys(df["lang"]))

    for m in models:
        df_m = df[df["model"] == m]
        for l in langs:
            df_l = df_m[df_m["lang"] == l]
            if df_l.empty:
                continue

            # Group by p
            rates = sorted(df_l["p"].unique())
            for p in rates:
                sub = df_l[df_l["p"] == p]
                acc = float(sub["correct"].mean())
                acc_low, acc_high = bootstrap_ci(sub["correct"].astype(float), n_boot=1000, seed=42)
                mean_ratio = float(sub["ratio"].mean())

                records.append({
                    "model": m,
                    "lang": l,
                    "p": p,
                    "mean_fertility_ratio": mean_ratio,
                    "accuracy": acc,
                    "accuracy_ci_low": acc_low,
                    "accuracy_ci_high": acc_high,
                    "n_items": len(sub),
                })

    res_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"Wrote E4 analysis to {out_path}")

    # Compute overall slopes per (model, lang)
    print("\n=== E4 Fragmentation Slopes (Accuracy drop per unit of fertility) ===")
    slope_records = []
    for m in models:
        df_m = df[df["model"] == m]
        for l in langs:
            df_l = df_m[df_m["lang"] == l]
            if df_l.empty:
                continue
            x = df_l["ratio"].values
            y = df_l["correct"].astype(float).values
            slope, intercept, r2 = fit_slope(x, y)
            s_low, s_high = bootstrap_slope_ci(x, y, n_boot=1000, seed=42)
            slope_records.append({
                "model": m,
                "lang": l,
                "slope": slope,
                "slope_ci_low": s_low,
                "slope_ci_high": s_high,
                "r2": r2,
            })
            print(f"[{m}] {l:10s} | Slope: {slope:+.4f} (95% CI: [{s_low:+.4f}, {s_high:+.4f}]) | R^2: {r2:.3f}")

    slope_df = pd.DataFrame(slope_records)
    slope_out = out_path.replace(".csv", "_slopes.csv")
    slope_df.to_csv(slope_out, index=False)
    print(f"Wrote E4 slopes to {slope_out}")
    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--out", default="results/e4_analysis.csv")
    args = parser.parse_args()
    analyze_e4(in_path=args.input, out_path=args.out)
