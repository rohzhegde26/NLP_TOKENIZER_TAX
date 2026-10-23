"""Kaggle GPU runner for Tokenizer Tax Experiment E4.

Runs Qwen2.5-3B-Instruct (4-bit NF4 on T4 GPU) on Belebele (first 300 questions)
across rates p in {0.0, 0.1, 0.3, 0.5} for English, Hindi, Tamil, and Malayalam.
Outputs e4_results.csv and e4_analysis.csv to /kaggle/working/.
"""
import os
import subprocess
import sys

print("=== Setting up GPU environment ===")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers", "bitsandbytes", "accelerate", "datasets", "tiktoken", "statsmodels", "sentencepiece"
], check=True)

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

# Clone repository branch lane-c
repo_dir = "/kaggle/working/NLP_TOKENIZER_TAX"
if not os.path.exists(repo_dir):
    print("Cloning repository branch lane-c...")
    subprocess.run([
        "git", "clone", "-b", "lane-c",
        "https://github.com/rohzhegde26/NLP_TOKENIZER_TAX.git",
        repo_dir
    ], check=True)

sys.path.insert(0, os.path.join(repo_dir, "src"))

from tokentax.tokenizers import get_tokenizer
from tokentax.eval.resegment import resegment
from tokentax.eval.score import load_model, score
from tokentax.data import load_belebele
from tokentax.schemas import COLUMNS, check

os.chdir(repo_dir)

print("\n=== Loading Qwen2.5-3B-Instruct (4-bit on GPU) ===")
scorer = load_model("qwen2.5-3b")
tok = get_tokenizer("qwen2.5")

langs = ["eng_Latn", "hin_Deva", "tam_Taml", "mal_Mlym"]
rates = [0.0, 0.1, 0.3, 0.5]
n_questions = 300
seed = 0

rows = []
for l in langs:
    print(f"\n--- Loading Belebele for {l} ({n_questions} questions) ---")
    df_b = load_belebele(l, n=n_questions)
    print(f"Loaded {len(df_b)} questions for {l}.")

    for p in rates:
        print(f"Running lang={l}, p={p}...")
        for idx, row in df_b.iterrows():
            passage = str(row["passage"])
            canonical_ids = tok.encode(passage)
            n_can = len(canonical_ids)

            if p == 0.0:
                used_ids = list(canonical_ids)
            else:
                used_ids = resegment(tok, canonical_ids, p=p, seed=seed)

            n_used = len(used_ids)
            options = [str(row["opt1"]), str(row["opt2"]), str(row["opt3"]), str(row["opt4"])]
            pred = score(scorer, used_ids, str(row["question"]), options)
            gold = int(row["gold"])
            correct = bool(pred == gold)

            rows.append(["qwen2.5-3b", l, p, seed, str(row["qid"]), n_can, n_used, pred, gold, correct])

import pandas as pd
df_e4 = pd.DataFrame(rows, columns=COLUMNS["e4_results"])
check(df_e4, "e4_results")

out_e4 = "/kaggle/working/e4_results.csv"
df_e4.to_csv(out_e4, index=False)
print(f"\nSuccessfully wrote {len(df_e4)} rows to {out_e4}")

# Run analysis
print("\n=== Running E4 Analysis ===")
subprocess.run([
    sys.executable, "scripts/analyze_e4.py",
    "--input", out_e4,
    "--out", "/kaggle/working/e4_analysis.csv"
], check=True)

print("\n=== E4 GPU Run Completed Successfully! ===")
