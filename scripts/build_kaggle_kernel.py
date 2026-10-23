"""Build standalone kaggle_kernel/main.py with embedded tokentax package."""
import base64
import io
import os
import zipfile

# 1. Zip all src/tokentax/*.py files
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk("src"):
        for f in files:
            if f.endswith(".py"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, "src")
                zf.write(full_path, rel_path)

b64 = base64.b64encode(buf.getvalue()).decode("ascii")

# 2. Generate standalone main.py
main_code = f'''"""Kaggle GPU runner for Tokenizer Tax Experiment E4.

Runs Qwen2.5-3B-Instruct (4-bit NF4 on Tesla T4 GPU) on Belebele (first 300 questions)
across rates p in {{0.0, 0.1, 0.3, 0.5}} for English, Hindi, Tamil, and Malayalam.
Outputs e4_results.csv, e4_analysis.csv, and e4_analysis_slopes.csv to /kaggle/working/.
"""
import base64
import io
import os
import subprocess
import sys
import zipfile

print("=== Setting up GPU environment ===")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers", "bitsandbytes", "accelerate", "datasets", "tiktoken", "statsmodels", "sentencepiece"
], check=True)

import torch
print(f"CUDA available: {{torch.cuda.is_available()}}")
if torch.cuda.is_available():
    print(f"Device: {{torch.cuda.get_device_name(0)}}")

# Extract embedded tokentax package
TOKENTAX_ZIP_B64 = "{b64}"
extract_dir = "/kaggle/working/src"
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(TOKENTAX_ZIP_B64))) as zf:
    zf.extractall(extract_dir)

sys.path.insert(0, extract_dir)

import numpy as np
import pandas as pd
from tqdm import tqdm

from tokentax.data import load_belebele
from tokentax.eval.resegment import resegment
from tokentax.eval.score import load_model, score
from tokentax.schemas import COLUMNS, check
from tokentax.stats import bootstrap_ci, bootstrap_slope_ci, fit_slope
from tokentax.tokenizers import get_tokenizer

print("\\n=== Loading Qwen2.5-3B-Instruct in 4-bit on GPU ===")
scorer = load_model("qwen2.5-3b")
tok = get_tokenizer("qwen2.5")

langs = ["eng_Latn", "hin_Deva", "tam_Taml", "mal_Mlym"]
rates = [0.0, 0.1, 0.3, 0.5]
n_questions = 300
seed = 0

rows = []
for l in langs:
    print(f"\\n--- Loading Belebele for {{l}} ({{n_questions}} questions) ---")
    df_b = load_belebele(l, n=n_questions)
    print(f"Loaded {{len(df_b)}} questions for {{l}}.")

    for p in rates:
        print(f"Running lang={{l}}, p={{p}}...")
        for idx, row in tqdm(df_b.iterrows(), total=len(df_b), desc=f"{{l}} p={{p}}"):
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

df_e4 = pd.DataFrame(rows, columns=COLUMNS["e4_results"])
check(df_e4, "e4_results")

out_e4 = "/kaggle/working/e4_results.csv"
df_e4.to_csv(out_e4, index=False)
print(f"\\nSuccessfully wrote {{len(df_e4)}} rows to {{out_e4}}")

# --- Analysis & Slopes ---
print("\\n=== Computing E4 Analysis and Slopes ===")
df_e4["ratio"] = df_e4["n_tokens_used"] / df_e4["n_tokens_canonical"]

analysis_records = []
slope_records = []

for l in langs:
    df_l = df_e4[df_e4["lang"] == l]
    for p in rates:
        sub = df_l[df_l["p"] == p]
        acc = float(sub["correct"].mean())
        acc_low, acc_high = bootstrap_ci(sub["correct"].astype(float), n_boot=1000, seed=42)
        mean_ratio = float(sub["ratio"].mean())
        analysis_records.append({{
            "model": "qwen2.5-3b",
            "lang": l,
            "p": p,
            "mean_fertility_ratio": mean_ratio,
            "accuracy": acc,
            "accuracy_ci_low": acc_low,
            "accuracy_ci_high": acc_high,
            "n_items": len(sub),
        }})

    x = df_l["ratio"].values
    y = df_l["correct"].astype(float).values
    slope, intercept, r2 = fit_slope(x, y)
    s_low, s_high = bootstrap_slope_ci(x, y, n_boot=1000, seed=42)
    slope_records.append({{
        "model": "qwen2.5-3b",
        "lang": l,
        "slope": slope,
        "slope_ci_low": s_low,
        "slope_ci_high": s_high,
        "r2": r2,
    }})
    print(f"[qwen2.5-3b] {{l:10s}} | Slope: {{slope:+.4f}} (95% CI: [{{s_low:+.4f}}, {{s_high:+.4f}}]) | R^2: {{r2:.3f}}")

out_analysis = "/kaggle/working/e4_analysis.csv"
out_slopes = "/kaggle/working/e4_analysis_slopes.csv"
pd.DataFrame(analysis_records).to_csv(out_analysis, index=False)
pd.DataFrame(slope_records).to_csv(out_slopes, index=False)
print(f"Wrote analysis to {{out_analysis}}")
print(f"Wrote slopes to {{out_slopes}}")

print("\\n=== E4 GPU Run Completed Successfully! ===")
'''

os.makedirs("kaggle_kernel", exist_ok=True)
with open("kaggle_kernel/main.py", "w", encoding="utf-8") as f:
    f.write(main_code)
print("Successfully generated standalone kaggle_kernel/main.py!")
