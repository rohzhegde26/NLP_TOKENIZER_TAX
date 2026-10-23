"""One shared figure style: fonts, colours, language order."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .langs import AUDIT_LANGS, LANG_NAMES

LANG_ORDER = AUDIT_LANGS
TOKENIZER_LABELS = {"cl100k": "cl100k (GPT-4)", "o200k": "o200k (GPT-4o)", "llama3": "Llama 3",
                    "qwen2.5": "Qwen2.5", "gemma2": "Gemma 2", "xlmr": "XLM-R", "sarvam1": "Sarvam-1"}
CONDITION_COLORS = {"full": "#2a6f97", "eqtok_1.0": "#e9a23b", "eqtok_0.5": "#c8553d"}
LANG_COLORS = {"eng_Latn": "#222222", "hin_Deva": "#2a6f97", "ben_Beng": "#6a994e", "tam_Taml": "#e9a23b",
               "tel_Telu": "#7b2cbf", "kan_Knda": "#bc6c25", "mal_Mlym": "#c8553d"}


def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
    })


def lang_label(code: str) -> str:
    return LANG_NAMES.get(code, code)


def ordered(langs):
    s = set(langs)
    return [l for l in LANG_ORDER if l in s]


def save(fig, name: str, outdir: str = "figures"):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{outdir}/{name}.{ext}")
    plt.close(fig)
    print(f"saved {outdir}/{name}.png/.pdf")
