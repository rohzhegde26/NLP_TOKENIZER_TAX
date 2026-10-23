"""Result-file schemas. Never rename a column after Day 1."""

RESULTS_DIR = "results"

COLUMNS = {
    "e1_intrinsic": ["tokenizer", "lang", "sent_id", "n_tokens", "n_words", "n_chars", "n_bytes", "n_single_byte"],
    "e2_cost": ["tokenizer", "lang", "tax_vs_eng", "usd_per_1k_eng_words", "inr_per_1k_eng_words",
                "pct_context_lost_128k", "price_snapshot_date"],
    "e3_results": ["model", "lang", "condition", "budget_tokens", "qid", "passage_tokens_full",
                   "passage_tokens_used", "answer_kept", "pred", "gold", "correct"],
    "e4_results": ["model", "lang", "p", "seed", "qid", "n_tokens_canonical", "n_tokens_used", "pred", "gold", "correct"],
}

CONDITIONS = ["full", "eqtok_1.0", "eqtok_0.5"]
E4_RATES = [0.0, 0.1, 0.3, 0.5]
E4_LANGS = ["eng_Latn", "hin_Deva", "tam_Taml", "mal_Mlym"]
MODELS = ["qwen2.5-3b", "gemma2-2b"]


def path(name: str) -> str:
    return f"{RESULTS_DIR}/{name}.csv"


def check(df, name: str) -> None:
    """Raise if df's columns differ from the contract."""
    if list(df.columns) != COLUMNS[name]:
        raise ValueError(f"{name}: columns {list(df.columns)} != contract {COLUMNS[name]}")


def load(name: str, results_dir: str = RESULTS_DIR):
    """Read a result CSV; fall back to results/fake/ (with a warning) until the real file exists."""
    import os
    import warnings

    import pandas as pd

    real = f"{results_dir}/{name}.csv"
    if os.path.exists(real):
        return pd.read_csv(real)
    fake = f"{results_dir}/fake/{name}.csv"
    warnings.warn(f"{real} missing; using FAKE data from {fake}")
    return pd.read_csv(fake)
