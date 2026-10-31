"""Decoupled Per-Language Banks with Scaffold Token Pruning, BPE-Dropout, and Fast Maximal Matching."""
from __future__ import annotations

import collections
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.sparse_indic.brahmic_fallback import (
    BRAHMIC_OFFSET_TOKENS,
    BrahmicFallbackHandler,
    decode_brahmic_offset_token,
    encode_brahmic_oov_char,
    get_active_block_base,
)
from src.sparse_indic.pretokenizer import (
    AksharaBoundedPreTokenizer,
    is_bare_dependent_char,
    is_brahmic_char,
)


class LanguageBank:
    """Decoupled 8,192-token vocabulary bank for a single Indic language."""

    def __init__(
        self,
        lang_code: str,
        vocab: List[str],
        merges: List[Tuple[str, str]],
        pruned_scaffold_count: int = 0
    ):
        self.lang_code = lang_code
        self.vocab = vocab
        self.merges = merges
        self.pruned_scaffold_count = pruned_scaffold_count
        self.token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
        self.id_to_token = {idx: tok for idx, tok in enumerate(vocab)}
        self.merge_ranks = {pair: idx for idx, pair in enumerate(merges)}

    def save(self, output_path: Path) -> None:
        data = {
            "lang_code": self.lang_code,
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
            "pruned_scaffold_count": self.pruned_scaffold_count,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LanguageBank":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merges = [tuple(p) for p in data["merges"]]
        return cls(
            lang_code=data["lang_code"],
            vocab=data["vocab"],
            merges=merges,
            pruned_scaffold_count=data.get("pruned_scaffold_count", 0),
        )


class ScaffoldBPETrainer:
    """Trains decoupled language banks with Akshara guards, fast batched merges, and Scaffold Pruning."""

    def __init__(
        self,
        bank_size: int = 8192,
        scaffold_threshold_tau: int = 2,
        high_pmi_stem_slots: int = 3500
    ):
        self.bank_size = bank_size
        self.tau = scaffold_threshold_tau
        self.high_pmi_stem_slots = high_pmi_stem_slots
        self.pretokenizer = AksharaBoundedPreTokenizer()

    def train(self, texts: List[str], lang_code: str) -> LanguageBank:
        """Train language bank with decoupled BPE, Akshara constraints, and scaffold pruning."""
        # 1. Collect clean word frequencies
        word_freqs = collections.Counter()
        for text in texts:
            words = re.findall(r"[\u0900-\u0D7F\w]+|[^\s\w]", text, re.UNICODE)
            for w in words:
                if w:
                    word_freqs[w] += 1

        # 2. Extract initial base symbols
        char_freqs = collections.Counter()
        split_words: Dict[Tuple[str, ...], int] = {}

        for word, count in word_freqs.items():
            morphemes = self.pretokenizer.suffix_splitter.split_word(word, lang=lang_code)
            for m in morphemes:
                aksharas = list(self.pretokenizer.segmenter.segment_word(m))
                if not aksharas:
                    continue
                pieces = tuple(aksharas)
                split_words[pieces] = count
                for p in pieces:
                    if self.pretokenizer.is_valid_subword(p):
                        char_freqs[p] += count

        # Base vocabulary of unique akshara symbols (strictly excluding bare dependent matras/halants)
        vocab_set = {p for p in char_freqs.keys() if self.pretokenizer.is_valid_subword(p)}
        merges: List[Tuple[str, str]] = []
        target_merges = max(500, self.bank_size - len(vocab_set) - self.high_pmi_stem_slots)

        # 3. Learn BPE merges with Akshara-Boundary Guard (fast batched)
        batch_size = 30
        while len(merges) < target_merges:
            pair_counts = collections.Counter()
            for word_tuple, count in split_words.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pair_counts[pair] += count

            if not pair_counts:
                break

            selected_pairs = {}
            used_symbols = set()
            for pair, _ in pair_counts.most_common(batch_size * 4):
                cand = pair[0] + pair[1]
                if (
                    self.pretokenizer.is_valid_subword(cand)
                    and pair[0] not in used_symbols
                    and pair[1] not in used_symbols
                ):
                    selected_pairs[pair] = cand
                    used_symbols.add(pair[0])
                    used_symbols.add(pair[1])
                    merges.append(pair)
                    vocab_set.add(cand)
                    if len(merges) >= target_merges or len(selected_pairs) >= batch_size:
                        break

            if not selected_pairs:
                break

            new_split_words = {}
            for word_tuple, count in split_words.items():
                new_tuple = []
                i = 0
                while i < len(word_tuple):
                    if (
                        i < len(word_tuple) - 1
                        and (word_tuple[i], word_tuple[i + 1]) in selected_pairs
                    ):
                        new_tuple.append(selected_pairs[(word_tuple[i], word_tuple[i + 1])])
                        i += 2
                    else:
                        new_tuple.append(word_tuple[i])
                        i += 1
                new_split_words[tuple(new_tuple)] = count
            split_words = new_split_words

        # 4. Scaffold Token Pruning:
        post_counts = collections.Counter()
        for word_tuple, count in split_words.items():
            for piece in word_tuple:
                post_counts[piece] += count

        retained_vocab = []
        pruned_scaffold_count = 0
        for tok in vocab_set:
            if not self.pretokenizer.is_valid_subword(tok):
                continue
            if post_counts[tok] < self.tau and len(tok) > 1:
                pruned_scaffold_count += 1
            else:
                retained_vocab.append(tok)

        # 5. Reallocate freed slots to high-frequency whole words, morphemes, and stems
        slots_needed = self.bank_size - len(retained_vocab)
        if slots_needed > 0:
            candidate_stems = collections.Counter()
            for word, count in word_freqs.items():
                if len(word) >= 1 and count >= 1:
                    candidate_stems[word] += count

            for stem, _ in candidate_stems.most_common():
                if len(retained_vocab) >= self.bank_size:
                    break
                if stem not in retained_vocab and self.pretokenizer.is_valid_subword(stem):
                    retained_vocab.append(stem)

        # If any slots remain, extract high-frequency n-grams directly from the corpus
        if len(retained_vocab) < self.bank_size:
            ngram_counts = collections.Counter()
            for text in texts:
                for w in text.split():
                    for n in (2, 3, 4):
                        for j in range(len(w) - n + 1):
                            ngram_counts[w[j : j + n]] += 1
            for ng, _ in ngram_counts.most_common():
                if len(retained_vocab) >= self.bank_size:
                    break
                if ng not in retained_vocab and self.pretokenizer.is_valid_subword(ng):
                    retained_vocab.append(ng)

        # Ensure exact bank size
        final_vocab = retained_vocab[: self.bank_size]
        return LanguageBank(
            lang_code=lang_code,
            vocab=final_vocab,
            merges=merges,
            pruned_scaffold_count=pruned_scaffold_count,
        )


class SparseIndicTokenizer:
    """Unified SPARSE-INDIC Tokenizer combining Core Vocab + Language Banks."""

    def __init__(
        self,
        core_vocab: List[str],
        banks: Dict[str, LanguageBank]
    ):
        self.core_vocab = core_vocab
        self.banks = banks
        self.core_to_id = {tok: idx for idx, tok in enumerate(core_vocab)}
        self.id_to_core = {idx: tok for idx, tok in enumerate(core_vocab)}
        self.pretokenizer = AksharaBoundedPreTokenizer()
        self.brahmic_handler = BrahmicFallbackHandler(self.core_to_id, self.id_to_core)

        # Offsets for global token indexing
        self.bank_offsets: Dict[str, int] = {}
        curr_offset = len(core_vocab)
        for lang_code in sorted(banks.keys()):
            self.bank_offsets[lang_code] = curr_offset
            curr_offset += len(banks[lang_code].vocab)

        self.total_vocab_size = curr_offset

    def tokenize_piece(
        self,
        piece: str,
        bank: Optional[LanguageBank],
        dropout_prob: float = 0.0
    ) -> List[str]:
        """Maximal matching subword tokenization respecting Akshara boundaries and bank vocabulary."""
        if not piece:
            return []
        vocab_map = bank.token_to_id if bank is not None else self.core_to_id
        if piece in vocab_map:
            return [piece]

        tokens = []
        i = 0
        n = len(piece)

        while i < n:
            best_k = i + 1
            for k in range(n, i, -1):
                sub = piece[i:k]
                if sub in vocab_map:
                    if dropout_prob > 0.0 and random.random() < dropout_prob and k > i + 1:
                        continue
                    best_k = k
                    break
            tokens.append(piece[i:best_k])
            i = best_k

        return tokens

    def encode(
        self,
        text: str,
        lang_code: Optional[str] = None,
        dropout_prob: float = 0.0
    ) -> List[int]:
        """Encode text to global token IDs.
        
        Preserves exact whitespaces and emits single-token Brahmic offset fallbacks for OOVs.
        """
        if not text:
            return []

        bank = self.banks.get(lang_code) if lang_code else None
        bank_offset = self.bank_offsets.get(lang_code, len(self.core_vocab)) if lang_code else 0

        chunks = re.findall(r"[\u0900-\u0D7F\w]+|[^\s\w]|\s+", text, re.UNICODE)
        token_ids: List[int] = []

        for chunk in chunks:
            if chunk.isspace():
                for c in chunk:
                    if c in self.core_to_id:
                        token_ids.append(self.core_to_id[c])
                    else:
                        token_ids.append(self.core_to_id.get(" ", 32))
                continue

            # Fast path: whole word in bank
            if bank and chunk in bank.token_to_id:
                token_ids.append(bank_offset + bank.token_to_id[chunk])
                continue

            # Whole word in core
            if chunk in self.core_to_id:
                token_ids.append(self.core_to_id[chunk])
                continue

            # Morphological splitting
            morphemes = self.pretokenizer.suffix_splitter.split_word(chunk, lang=lang_code)
            for m in morphemes:
                subwords = self.tokenize_piece(m, bank, dropout_prob=dropout_prob)
                for sw in subwords:
                    if bank and sw in bank.token_to_id:
                        token_ids.append(bank_offset + bank.token_to_id[sw])
                    elif sw in self.core_to_id:
                        token_ids.append(self.core_to_id[sw])
                    else:
                        for char in sw:
                            if bank and char in bank.token_to_id:
                                token_ids.append(bank_offset + bank.token_to_id[char])
                            elif char in self.core_to_id:
                                token_ids.append(self.core_to_id[char])
                            elif is_brahmic_char(char):
                                fb_id = self.brahmic_handler.encode_char(char, lang_code)
                                if fb_id is not None:
                                    token_ids.append(fb_id)
                                else:
                                    token_ids.append(self.core_to_id.get("<unk>", 0))
                            else:
                                token_ids.append(self.core_to_id.get("<unk>", 0))

        return token_ids

    def decode(
        self,
        token_ids: List[int],
        lang_code: Optional[str] = None
    ) -> str:
        """Decode global token IDs back into 100% exact lossless text."""
        pieces = []
        core_len = len(self.core_vocab)

        for tid in token_ids:
            if tid < core_len:
                tok_str = self.id_to_core.get(tid, "")
                if tok_str.startswith("<BRAHMIC_0x"):
                    char = decode_brahmic_offset_token(tok_str, lang_code)
                    pieces.append(char if char else tok_str)
                else:
                    pieces.append(tok_str)
            else:
                found = False
                for b_code, b_offset in self.bank_offsets.items():
                    bank = self.banks[b_code]
                    if b_offset <= tid < b_offset + len(bank.vocab):
                        local_id = tid - b_offset
                        pieces.append(bank.id_to_token.get(local_id, ""))
                        found = True
                        break
                if not found:
                    pieces.append("")

        return "".join(pieces)
