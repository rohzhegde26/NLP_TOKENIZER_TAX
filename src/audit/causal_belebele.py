"""Causal Downstream Disentanglement on Belebele: Equal-Token vs. Equal-Content."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def format_belebele_prompt(
    passage: str,
    question: str,
    options: List[str]
) -> str:
    """Format prompt context ending with 'Correct Answer: ('."""
    context = (
        f"Read the following passage and answer the multiple-choice question.\n\n"
        f"Passage: {passage}\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n"
        f"(1) {options[0]}\n"
        f"(2) {options[1]}\n"
        f"(3) {options[2]}\n"
        f"(4) {options[3]}\n\n"
        f"Correct Answer: ("
    )
    return context


def truncate_text_to_tokens(
    text: str,
    tokenizer: Any,
    max_tokens: int
) -> str:
    """Truncate text so its token representation does not exceed max_tokens."""
    if hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            tokens = tokenizer.encode(text)

        if len(tokens) <= max_tokens:
            return text
        truncated_ids = tokens[:max_tokens]
        try:
            return tokenizer.decode(truncated_ids, skip_special_tokens=True)
        except TypeError:
            return tokenizer.decode(truncated_ids)
    else:
        words = text.split()
        return " ".join(words[:max(1, int(max_tokens / 1.5))])


class BelebeleCausalEvaluator:
    """Fast, vectorized harness to evaluate LLMs on Belebele under Equal-Token vs. Equal-Content budgets."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.choice_ids = None

    def load_model(self) -> None:
        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"Loading {self.model_name} onto {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()

            # Pre-extract single token IDs for choices 1, 2, 3, 4
            self.choice_ids = [
                self.tokenizer(str(i), add_special_tokens=False).input_ids[0]
                for i in range(1, 5)
            ]
            print(f"Choice token IDs (1..4): {self.choice_ids}")

    def evaluate_batch_prompts(self, prompts: List[str]) -> List[int]:
        """Given a list of prompts ending with 'Correct Answer: (', predict choice (1..4) in a single pass."""
        self.load_model()
        preds = []
        # Process in mini-batches
        batch_size = 8 if self.device == "cuda" else 4
        for idx in range(0, len(prompts), batch_size):
            batch_texts = prompts[idx : idx + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                # Next token logits at final non-padded position
                attention_mask = encoded["attention_mask"]
                seq_lens = attention_mask.sum(dim=1) - 1

                for b in range(len(batch_texts)):
                    final_pos = seq_lens[b].item()
                    logits = outputs.logits[b, final_pos, :]
                    choice_logits = [logits[cid].item() for cid in self.choice_ids]
                    pred = int(np.argmax(choice_logits)) + 1
                    preds.append(pred)

        return preds

    def run_causal_comparison(
        self,
        lang_data: Dict[str, List[Dict[str, Any]]],
        parity_map: Dict[str, float],
        fixed_budget: int = 96,
        sample_limit: Optional[int] = 30
    ) -> pd.DataFrame:
        """Run paired evaluation across Condition A (Equal-Token) and Condition B (Equal-Content)."""
        self.load_model()
        results = []

        for lang, items in lang_data.items():
            eval_items = items[:sample_limit] if sample_limit else items
            parity = max(0.8, parity_map.get(lang, 1.0))
            scaled_budget = int(math.ceil(fixed_budget * parity))

            prompts_a = []
            prompts_b = []
            golds = []

            for item in eval_items:
                gold = item["correct_answer_num"]
                passage = item["flores_passage"]
                question = item["question"]
                options = [
                    item["mc_answer1"],
                    item["mc_answer2"],
                    item["mc_answer3"],
                    item["mc_answer4"],
                ]
                golds.append(gold)

                # Condition A: Truncated to fixed token budget
                pass_a = truncate_text_to_tokens(passage, self.tokenizer, fixed_budget)
                prompts_a.append(format_belebele_prompt(pass_a, question, options))

                # Condition B: Truncated to scaled token budget (Equal-Content)
                pass_b = truncate_text_to_tokens(passage, self.tokenizer, scaled_budget)
                prompts_b.append(format_belebele_prompt(pass_b, question, options))

            preds_a = self.evaluate_batch_prompts(prompts_a)
            preds_b = self.evaluate_batch_prompts(prompts_b)

            correct_a = sum(1 for p, g in zip(preds_a, golds) if p == g)
            correct_b = sum(1 for p, g in zip(preds_b, golds) if p == g)
            total = len(golds)

            acc_a = (correct_a / total) * 100.0 if total > 0 else 0.0
            acc_b = (correct_b / total) * 100.0 if total > 0 else 0.0
            delta_truncation = acc_b - acc_a

            print(f"  {lang}: Eq-Tok Acc={acc_a:.1f}% | Eq-Cont Acc={acc_b:.1f}% | Delta={delta_truncation:+.1f}%")

            results.append({
                "language": lang,
                "parity_ratio": round(parity, 3),
                "fixed_budget_tokens": fixed_budget,
                "scaled_budget_tokens": scaled_budget,
                "acc_equal_token_pct": round(acc_a, 2),
                "acc_equal_content_pct": round(acc_b, 2),
                "delta_acc_truncation_pct": round(delta_truncation, 2),
                "sample_count": total,
            })

        df = pd.DataFrame(results)

        # Baseline English accuracy under Condition B
        eng_row = df[df["language"] == "eng_Latn"]
        eng_acc = eng_row["acc_equal_content_pct"].iloc[0] if not eng_row.empty else 60.0

        df["delta_acc_capability_pct"] = (eng_acc - df["acc_equal_content_pct"]).clip(lower=0.0).round(2)
        return df


def bootstrap_significance(
    diffs: List[float],
    n_bootstraps: int = 1000,
    alpha: float = 0.05
) -> Dict[str, float]:
    """Compute paired bootstrap confidence intervals and empirical p-value for the truncation gap."""
    arr = np.array(diffs, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0}

    boot_means = []
    n = len(arr)
    rng = np.random.default_rng(42)
    for _ in range(n_bootstraps):
        resample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(resample))

    mean_val = float(np.mean(arr))
    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    p_value = float(np.mean(np.array(boot_means) <= 0.0))

    return {
        "mean": round(mean_val, 3),
        "ci_lower": round(ci_lower, 3),
        "ci_upper": round(ci_upper, 3),
        "p_value": round(p_value, 4),
    }
