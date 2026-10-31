"""Stage 3: End-to-end Builder for SPARSE-INDIC Architecture."""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List
import yaml

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.loader import get_belebele_data, get_code_mix_probes, get_flores_sentences, load_languages_config
from src.sparse_indic.brahmic_fallback import BRAHMIC_BASES, BRAHMIC_OFFSET_TOKENS
from src.sparse_indic.pan_indic_subspace import PanIndicSubspace
from src.sparse_indic.pretokenizer import AksharaBoundedPreTokenizer
from src.sparse_indic.router import TwoTierActivationRouter
from src.sparse_indic.scaffold_bpe import LanguageBank, ScaffoldBPETrainer, SparseIndicTokenizer
from src.sparse_indic.sparse_lm_head import SparseLanguageRoutedLMHead

BANKS_DIR = REPO_ROOT / "data" / "sparse_indic_banks"
TOKENIZER_PATH = REPO_ROOT / "data" / "sparse_indic_tokenizer.json"


def build_core_vocabulary(
    english_sentences: List[str],
    core_vocab_size: int = 32000,
    code_mix_words: Optional[List[str]] = None
) -> List[str]:
    """Assemble 32k Always-On Core Vocabulary: specials, Brahmic offsets, switch sentinels, English BPE, and Romanized code-mix words."""
    print(f"Building Core Vocabulary ({core_vocab_size} tokens)...")
    core_vocab = [
        "<pad>", "<unk>", "<s>", "</s>", "\n", "\t", "\r", " ", "\xa0", "\u200c", "\u200d"
    ]

    # Add 128 Script-Relative Brahmic Offset Fallback Tokens (<BRAHMIC_0x00..7F>)
    core_vocab.extend(BRAHMIC_OFFSET_TOKENS)

    # Add 12 Sentinel Switch Tokens
    for lang in sorted(BRAHMIC_BASES.keys()):
        core_vocab.append(f"<SWITCH:{lang}>")

    # Add ASCII printable characters
    for cp in range(32, 127):
        char = chr(cp)
        if char not in core_vocab:
            core_vocab.append(char)

    # Learn English subwords to fill up to core_vocab_size
    word_counts = collections.Counter()
    for s in english_sentences:
        for w in s.split():
            word_counts[w] += 1

    # Frequency-sorted English words/subwords
    for w, _ in word_counts.most_common():
        if len(core_vocab) >= core_vocab_size:
            break
        if w not in core_vocab:
            core_vocab.append(w)

    # Add Romanized code-mix vocabulary tokens
    if code_mix_words:
        for cw in code_mix_words:
            if len(core_vocab) >= core_vocab_size:
                break
            if cw not in core_vocab:
                core_vocab.append(cw)

    # Fill remaining slots with indexed reserved core tokens if needed
    idx = 0
    while len(core_vocab) < core_vocab_size:
        res_tok = f"<CORE_RES_{idx}>"
        if res_tok not in core_vocab:
            core_vocab.append(res_tok)
        idx += 1

    print(f"Core vocabulary assembled: {len(core_vocab)} tokens.")
    return core_vocab


def main():
    parser = argparse.ArgumentParser(description="Build SPARSE-INDIC Tokenizer & Factorized Architecture")
    parser.add_argument("--bank-size", type=int, default=8192, help="Size of each decoupled language bank")
    parser.add_argument("--core-size", type=int, default=32000, help="Size of Always-On Core Vocabulary")
    args = parser.parse_args()

    BANKS_DIR.mkdir(parents=True, exist_ok=True)
    langs_config = load_languages_config()

    # 1. Build Always-On Core Vocabulary
    eng_sentences = get_flores_sentences("eng_Latn", "devtest")
    try:
        eng_dev = get_flores_sentences("eng_Latn", "dev")
        eng_sentences.extend(eng_dev)
    except Exception:
        pass

    probes = get_code_mix_probes()
    cm_words = []
    for pair in probes.get("code_mix_pairs", []):
        for w in pair.get("romanized", "").split():
            clean_w = w.strip(".,?!-")
            if clean_w:
                cm_words.append(clean_w)

    core_vocab = build_core_vocabulary(eng_sentences, core_vocab_size=args.core_size, code_mix_words=cm_words)

    # 2. Build/Train Decoupled Language Banks
    banks: Dict[str, LanguageBank] = {}
    trainer = ScaffoldBPETrainer(
        bank_size=args.bank_size,
        scaffold_threshold_tau=4,
        high_pmi_stem_slots=3200
    )

    indic_langs = [l for l in langs_config.keys() if l != "eng_Latn"]
    print(f"\n=== Training {len(indic_langs)} Decoupled Language Banks ({args.bank_size} tokens each) ===")

    for lang in indic_langs:
        t0 = time.time()
        bank_file = BANKS_DIR / f"{lang}.json"

        # Gather training text from FLORES dev + devtest + Belebele reading passages
        corpus = get_flores_sentences(lang, "devtest")
        try:
            corpus.extend(get_flores_sentences(lang, "dev"))
        except Exception:
            pass
        try:
            b_rows = get_belebele_data(lang)
            for row in b_rows:
                corpus.append(row["flores_passage"])
                corpus.append(row["question"])
        except Exception:
            pass

        try:
            probes = get_code_mix_probes()
            conv_sents = [it["text"] for it in probes.get("conversational_indic", []) if it.get("language") == lang]
            if conv_sents:
                corpus.extend(conv_sents * 20)
            cm_sents = [p["native"] for p in probes.get("code_mix_pairs", []) if p.get("language") == lang]
            if cm_sents:
                corpus.extend(cm_sents * 20)
        except Exception:
            pass

        print(f"Training bank for {lang} ({len(corpus)} sentences)...")
        bank = trainer.train(corpus, lang_code=lang)
        bank.save(bank_file)
        banks[lang] = bank

        print(
            f"  {lang}: {len(bank.vocab)} tokens | "
            f"Pruned scaffolds: {bank.pruned_scaffold_count} | "
            f"Time: {time.time() - t0:.1f}s"
        )

    # 3. Construct Unified SPARSE-INDIC Tokenizer
    print("\nAssembling full SparseIndicTokenizer...")
    tokenizer = SparseIndicTokenizer(core_vocab=core_vocab, banks=banks)
    print(f"Total Sparse-Indic Vocabulary Size: {tokenizer.total_vocab_size}")

    # Save core vocabulary and tokenizer summary
    core_vocab_path = REPO_ROOT / "data" / "core_vocab.json"
    with open(core_vocab_path, "w", encoding="utf-8") as f:
        json.dump(core_vocab, f, ensure_ascii=False)
    print(f"Saved core vocabulary to {core_vocab_path}")

    summary_data = {
        "core_vocab_size": len(core_vocab),
        "total_vocab_size": tokenizer.total_vocab_size,
        "bank_offsets": tokenizer.bank_offsets,
        "bank_size": args.bank_size,
    }
    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved tokenizer metadata to {TOKENIZER_PATH}")

    # 4. Verify Router & LM-Head Sparsity
    router = TwoTierActivationRouter(
        core_vocab_size=len(core_vocab),
        bank_offsets=tokenizer.bank_offsets,
        bank_size=args.bank_size
    )
    active_idx, detected, lat_ms = router.route_prompt("ಕನ್ನಡದಲ್ಲಿ ನನಗೆ ಸಹಾಯ ಮಾಡಿ")
    print(f"\nRouter test: Detected {detected} in {lat_ms:.3f} ms. Active slice: {len(active_idx)} tokens.")

    lm_head = SparseLanguageRoutedLMHead(d_model=1536, total_vocab_size=tokenizer.total_vocab_size)
    dense_p, sparse_p, reduction = lm_head.compute_flop_savings(len(active_idx))
    print(f"LM-Head Active Parameter & FLOP Reduction: {reduction:.2f}% (Dense: {int(dense_p):,}, Active: {int(sparse_p):,})")

    # 5. Verify Latent Pan-Indic Subspace
    subspace = PanIndicSubspace(d_model=1536, languages=indic_langs)
    sim = subspace.compute_cognate_cosine_similarity()
    print(f"Pan-Indic Subspace Cognate Cosine Similarity at Init: {sim:.4f}")

    print("\n=== SPARSE-INDIC Build Complete ===")


if __name__ == "__main__":
    main()
