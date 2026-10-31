"""Publication-ready visualizer: Heatmaps, Lorenz/Gini curves, Pareto frontiers, Causal plots."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_parity_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Cross-Lingual Tokenization Parity Ratio ($P_L$)"
) -> None:
    """Generate heatmap of parity ratio across tokenizers and languages."""
    pivot = df.pivot(index="tokenizer", columns="language", values="parity_ratio")
    plt.figure(figsize=(14, 6), dpi=300)
    sns.set_theme(style="white")
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Parity Ratio ($P_L$ rel. to English)"},
        linewidths=0.5
    )
    plt.title(title, fontsize=14, weight="bold", pad=12)
    plt.xlabel("Language (BCP-47)", fontsize=11, weight="bold")
    plt.ylabel("Tokenizer", fontsize=11, weight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_lorenz_gini_curves(
    tokenizer_parities: Dict[str, List[float]],
    output_path: Path,
    title: str = "Lorenz Curves of Cross-Lingual Tokenization Inequality"
) -> None:
    """Plot Lorenz curves and report Gini coefficient for each tokenizer."""
    plt.figure(figsize=(8, 7), dpi=300)
    sns.set_theme(style="ticks")

    # Line of perfect equality
    plt.plot([0, 1], [0, 1], "k--", label="Line of Perfect Equality (Gini = 0.0)", alpha=0.7)

    for tok_name, parities in tokenizer_parities.items():
        arr = np.sort(np.array(parities, dtype=np.float64))
        n = len(arr)
        cum_parities = np.cumsum(arr)
        cum_parities = np.insert(cum_parities, 0, 0)
        cum_parities /= cum_parities[-1]
        x_pts = np.linspace(0, 1, n + 1)

        # Gini calculation
        abs_diff = np.abs(np.subtract.outer(arr, arr))
        gini = float(np.sum(abs_diff) / (2.0 * n * np.sum(arr)))
        plt.plot(x_pts, cum_parities, label=f"{tok_name} (Gini = {gini:.3f})", lw=2)

    plt.title(title, fontsize=13, weight="bold")
    plt.xlabel("Cumulative Share of Indic Languages", fontsize=11)
    plt.ylabel("Cumulative Share of Token Footprint", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_causal_attribution(
    causal_df: pd.DataFrame,
    output_path: Path,
    title: str = "Causal Attribution: Truncation Erosion vs. Model Capability Gap"
) -> None:
    """Plot stacked or grouped bar chart comparing truncation loss vs intrinsic capability loss."""
    df_sorted = causal_df.sort_values(by="parity_ratio", ascending=False)
    langs = df_sorted["language"].tolist()
    trunc_loss = df_sorted["delta_acc_truncation_pct"].tolist()
    cap_gap = df_sorted["delta_acc_capability_pct"].tolist()

    x = np.arange(len(langs))
    width = 0.38

    plt.figure(figsize=(12, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    plt.bar(x - width/2, trunc_loss, width, label=r"$\Delta$Acc (Truncation Erosion: Eq-Content - Eq-Token)", color="#d95f02", alpha=0.85)
    plt.bar(x + width/2, cap_gap, width, label=r"$\Delta$Acc (Capability Gap: English - Eq-Content)", color="#7570b3", alpha=0.85)

    plt.title(title, fontsize=13, weight="bold")
    plt.xlabel("Language (Ranked by Parity Ratio)", fontsize=11)
    plt.ylabel("Accuracy Delta (%)", fontsize=11)
    plt.xticks(x, langs, rotation=45, ha="right")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pareto_frontier(
    points: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Pareto Frontier: Mean Indic Parity Ratio vs. Active Vocabulary VRAM"
) -> None:
    """Plot trade-off between parity efficiency and active memory footprint."""
    plt.figure(figsize=(9, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    for pt in points:
        name = pt["name"]
        x = pt["active_vocab_size"]
        y = pt["mean_parity"]
        color = pt.get("color", "blue")
        marker = pt.get("marker", "o")
        plt.scatter(x, y, s=150, color=color, marker=marker, label=name)
        plt.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(7, 7),
            fontsize=9,
            weight="bold"
        )

    plt.title(title, fontsize=13, weight="bold")
    plt.xlabel("Active GPU Vocabulary Size (Parameters)", fontsize=11)
    plt.ylabel("Mean Indic Parity Ratio ($P_L$, lower is better)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
