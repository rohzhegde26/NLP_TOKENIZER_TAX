"""Phase 1: Empirical Audit & Causal Disentanglement Runner."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.loader import get_belebele_data, get_flores_sentences, load_languages_config
from src.audit.context_erosion import compute_context_erosion, compute_monetary_tax
from src.audit.intrinsic_metrics import BaseTokenizerWrapper, audit_corpus_metrics
from src.audit.causal_belebele import BelebeleCausalEvaluator, bootstrap_significance
from src.visualization.plots import (
    plot_causal_attribution,
    plot_lorenz_gini_curves,
    plot_parity_heatmap,
)

CONFIG_DIR = REPO_ROOT / "configs"
RESULTS_TABLES = REPO_ROOT / "results" / "tables"
RESULTS_FIGURES = REPO_ROOT / "results" / "figures"


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 Intrinsic & Causal Audit")
    parser.add_argument("--sentence-limit", type=int, default=None, help="Limit FLORES sentences for fast smoke testing")
    parser.add_argument("--belebele-limit", type=int, default=25, help="Number of Belebele questions per language")
    parser.add_argument("--skip-causal", action="store_true", help="Skip causal Belebele evaluation")
    args = parser.parse_args()

    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_DIR / "languages.yaml", "r", encoding="utf-8") as f:
        langs_cfg = yaml.safe_load(f)["languages"]
    with open(CONFIG_DIR / "tokenizers.yaml", "r", encoding="utf-8") as f:
        toks_cfg = yaml.safe_load(f)["tokenizers"]

    target_langs = list(langs_cfg.keys())
    print(f"=== Phase 1: Auditing {len(toks_cfg)} tokenizers across {len(target_langs)} languages ===")

    # 1. Load FLORES-200 parallel sentences
    print("Loading FLORES-200 parallel devtest sentences...")
    corpus = {}
    for lang in target_langs:
        sents = get_flores_sentences(lang, split="devtest")
        if args.sentence_limit:
            sents = sents[: args.sentence_limit]
        corpus[lang] = sents
    print(f"Loaded {len(corpus['eng_Latn'])} sentences per language.")

    all_metrics_dfs = []
    summaries = []
    lorenz_data = {}

    for tok_key, tok_info in toks_cfg.items():
        print(f"\nProcessing tokenizer: {tok_info['name']} ({tok_info['identifier']}) ...")
        t0 = time.time()
        try:
            wrapper = BaseTokenizerWrapper(
                name=tok_info["name"],
                identifier=tok_info["identifier"],
                tok_type=tok_info["type"],
            )
        except Exception as e:
            print(f"Primary identifier failed for {tok_key}: {e}. Trying alternative...")
            alt = tok_info.get("alt_identifier")
            if not alt:
                raise e
            wrapper = BaseTokenizerWrapper(
                name=tok_info["name"],
                identifier=alt,
                tok_type=tok_info["type"],
            )

        df_metrics, summary = audit_corpus_metrics(corpus, wrapper, baseline_lang="eng_Latn")
        all_metrics_dfs.append(df_metrics)
        summaries.append(summary)

        indic_parities = df_metrics[df_metrics["language"] != "eng_Latn"]["parity_ratio"].tolist()
        lorenz_data[tok_info["name"]] = indic_parities

        print(
            f"  Mean Indic Parity: {summary['mean_indic_parity']:.3f}x | "
            f"Indic Gini: {summary['gini_indic']:.4f} | "
            f"Bytes/Tok: {summary['mean_bytes_per_token']:.2f} | "
            f"Elapsed: {time.time() - t0:.1f}s"
        )

    full_intrinsic_df = pd.concat(all_metrics_dfs, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    # Context Erosion & Monetary Tax
    erosion_df = compute_context_erosion(full_intrinsic_df)
    monetary_df = compute_monetary_tax(full_intrinsic_df, toks_cfg)

    # Save CSV tables
    intrinsic_path = RESULTS_TABLES / "phase1_intrinsic_tax.csv"
    summary_path = RESULTS_TABLES / "phase1_summary.csv"
    erosion_path = RESULTS_TABLES / "phase1_context_erosion.csv"
    monetary_path = RESULTS_TABLES / "phase1_monetary_tax.csv"

    full_intrinsic_df.to_csv(intrinsic_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    erosion_df.to_csv(erosion_path, index=False)
    monetary_df.to_csv(monetary_path, index=False)

    print(f"\nSaved intrinsic metrics to {intrinsic_path}")
    print(f"Saved summary metrics to {summary_path}")

    # Generate figures
    plot_parity_heatmap(
        full_intrinsic_df,
        RESULTS_FIGURES / "phase1_parity_heatmap.png"
    )
    plot_lorenz_gini_curves(
        lorenz_data,
        RESULTS_FIGURES / "phase1_lorenz_gini_curves.png"
    )
    print(f"Saved figures to {RESULTS_FIGURES}")

    # 2. Causal Belebele evaluation
    if not args.skip_causal:
        print("\n=== Phase 1.2: Causal Downstream Disentanglement on Belebele ===")
        belebele_data = {}
        for lang in target_langs:
            belebele_data[lang] = get_belebele_data(lang)

        # Use Qwen2.5 parity ratio map
        qwen_rows = full_intrinsic_df[full_intrinsic_df["tokenizer"].str.contains("Qwen")]
        parity_map = dict(zip(qwen_rows["language"], qwen_rows["parity_ratio"]))

        evaluator = BelebeleCausalEvaluator(model_name="Qwen/Qwen2.5-0.5B")
        causal_df = evaluator.run_causal_comparison(
            belebele_data,
            parity_map=parity_map,
            fixed_budget=96,
            sample_limit=args.belebele_limit
        )

        causal_path = RESULTS_TABLES / "phase1_belebele_causal.csv"
        causal_df.to_csv(causal_path, index=False)
        print(f"Saved causal Belebele results to {causal_path}")

        # Bootstrap analysis
        diffs = causal_df["delta_acc_truncation_pct"].tolist()
        boot_res = bootstrap_significance(diffs)
        print(
            f"Bootstrap 95% CI for Truncation Erosion: "
            f"[{boot_res['ci_lower']}%, {boot_res['ci_upper']}%], p = {boot_res['p_value']}"
        )

        plot_causal_attribution(
            causal_df,
            RESULTS_FIGURES / "phase1_causal_attribution.png"
        )
        print(f"Saved causal plot to {RESULTS_FIGURES / 'phase1_causal_attribution.png'}")

    print("\n=== Phase 1 Audit Finished Successfully ===")


if __name__ == "__main__":
    main()
