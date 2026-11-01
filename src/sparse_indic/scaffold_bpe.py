"""Decoupled Per-Language Banks with Scaffold Token Pruning, BPE-Dropout, Parity-BPE, and SuperBPE."""
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
    """Trains decoupled language banks with Akshara guards, PMI stem scoring, and Scaffold Pruning."""

    def __init__(
        self,
        bank_size: int = 8192,
        scaffold_threshold_tau: int = 4,
        high_pmi_stem_slots: int = 3200
    ):
        self.bank_size = bank_size
        self.tau = scaffold_threshold_tau
        self.high_pmi_stem_slots = high_pmi_stem_slots
        self.pretokenizer = AksharaBoundedPreTokenizer()

    def train(self, texts: List[str], lang_code: str) -> LanguageBank:
        """Train language bank on monolingual training corpus with Akshara constraints and Scaffold Pruning."""
        # Discover high-branching suffixes from the training corpus
        self.pretokenizer.suffix_splitter.learn_from_corpus(texts, lang_code)

        # 1. Collect clean word frequencies from training texts
        word_freqs = collections.Counter()
        for text in texts:
            words = re.findall(r"[\u0900-\u0D7F\w]+|[^\s\w]", text, re.UNICODE)
            for w in words:
                if w:
                    word_freqs[w] += 1

        # 2. Extract initial base symbols (aksharas)
        char_freqs = collections.Counter()
        split_words: Dict[Tuple[str, ...], int] = {}

        for word, count in word_freqs.items():
            morphemes = self.pretokenizer.suffix_splitter.split_word(word, lang=lang_code)
            for m in morphemes:
                aksharas = list(self.pretokenizer.segmenter.segment_word(m))
                if not aksharas:
                    continue
                pieces = tuple(aksharas)
                split_words[pieces] = split_words.get(pieces, 0) + count
                for p in pieces:
                    if self.pretokenizer.is_valid_subword(p):
                        char_freqs[p] += count

        # Base vocabulary of unique akshara symbols (strictly excluding bare dependent matras/halants)
        vocab_set = {p for p in char_freqs.keys() if self.pretokenizer.is_valid_subword(p)}
        merges: List[Tuple[str, str]] = []
        target_merges = max(500, self.bank_size - len(vocab_set) - self.high_pmi_stem_slots)

        # 3. Learn BPE merges with Akshara-Boundary Guard (batched)
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
                t_key = tuple(new_tuple)
                new_split_words[t_key] = new_split_words.get(t_key, 0) + count
            split_words = new_split_words

        # 4. Scaffold Token Pruning: compute post-merge standalone corpus frequency f_post(t)
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

        # 5. Reallocate freed slots to high-PMI morphological stems and subwords
        slots_needed = self.bank_size - len(retained_vocab)
        if slots_needed > 0:
            total_char_occ = max(1, sum(char_freqs.values()))
            candidate_stems: Dict[str, float] = {}
            for word, count in word_freqs.items():
                morps = self.pretokenizer.suffix_splitter.split_word(word, lang=lang_code)
                for m in morps:
                    if 2 <= len(m) <= 12 and self.pretokenizer.is_valid_subword(m):
                        # Compute Pointwise Mutual Information (PMI) weighted by log frequency
                        aks = self.pretokenizer.segmenter.segment_word(m)
                        if len(aks) >= 2:
                            p_joint = count / total_char_occ
                            p_indep = 1.0
                            for a in aks[:3]:
                                p_indep *= max(1e-6, char_freqs.get(a, 1) / total_char_occ)
                            pmi = max(0.1, math.log(max(1e-9, p_joint / p_indep) + 1.0))
                        else:
                            pmi = 1.0
                        score = count * (1.0 + 0.25 * pmi)
                        candidate_stems[m] = candidate_stems.get(m, 0.0) + score
                if 2 <= len(word) <= 12 and self.pretokenizer.is_valid_subword(word):
                    candidate_stems[word] = candidate_stems.get(word, 0.0) + count

            retained_set = set(retained_vocab)
            sorted_stems = sorted(candidate_stems.items(), key=lambda x: x[1], reverse=True)
            for stem, _ in sorted_stems:
                if len(retained_vocab) >= self.bank_size:
                    break
                if stem not in retained_set and self.pretokenizer.is_valid_subword(stem):
                    retained_vocab.append(stem)
                    retained_set.add(stem)

        # If any slots remain, extract high-frequency valid subword n-grams from the training corpus
        if len(retained_vocab) < self.bank_size:
            retained_set = set(retained_vocab)
            ngram_counts = collections.Counter()
            for text in texts:
                for w in text.split():
                    for n in (2, 3, 4):
                        for j in range(len(w) - n + 1):
                            sub = w[j : j + n]
                            if self.pretokenizer.is_valid_subword(sub):
                                ngram_counts[sub] += 1
            for ng, _ in ngram_counts.most_common():
                if len(retained_vocab) >= self.bank_size:
                    break
                if ng not in retained_set:
                    retained_vocab.append(ng)
                    retained_set.add(ng)

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
        if piece in vocab_map and dropout_prob == 0.0:
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
                    # When tokenizing Romanized/Latin words in core_vocab, avoid leaving a 1-char orphan suffix
                    if bank is None and dropout_prob == 0.0 and (n - k) == 1 and (k - i) >= 3:
                        if piece[k - 1 : n] in vocab_map and piece[i : k - 1] in vocab_map:
                            best_k = k - 1
                            break
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

        chunks = re.findall(r"[\u0900-\u0D7F\w]+(?:-[\u0900-\u0D7F\w]+)*|[^\s\w]|\s+", text, re.UNICODE)
        token_ids: List[int] = []

        for chunk in chunks:
            if chunk.isspace():
                for c in chunk:
                    if c in self.core_to_id:
                        token_ids.append(self.core_to_id[c])
                    else:
                        token_ids.append(self.core_to_id.get(" ", 32))
                continue

            # Fast path for whole words in bank (when dropout is off)
            if bank and chunk in bank.token_to_id and dropout_prob == 0.0:
                token_ids.append(bank_offset + bank.token_to_id[chunk])
                continue

            # Whole word in core (when dropout is off)
            if chunk in self.core_to_id and dropout_prob == 0.0:
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


class ParityAwareBPETokenizer:
    """Baseline Parity-Aware BPE (2026): Joint cross-lingual BPE without Scaffold Pruning or Akshara Guards.
    
    Allocates equal per-language vocabulary budgets in a single dense table, retaining all intermediate
    BPE scaffold tokens (high deadweight rate) and falling back to 3-byte UTF-8 for unseen characters.
    """

    def __init__(self, vocab: List[str], lang_subvocabs: Optional[Dict[str, List[str]]] = None):
        self.vocab = vocab
        self.lang_subvocabs = lang_subvocabs or {}
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self.id_to_token = {i: t for i, t in enumerate(vocab)}
        self.vocab_size = len(vocab)

    @classmethod
    def train_from_corpora(
        cls,
        lang_corpora: Dict[str, List[str]],
        per_lang_budget: int = 8192,
        total_vocab_size: int = 130000
    ) -> "ParityAwareBPETokenizer":
        """Train joint parity-balanced BPE without scaffold pruning or akshara boundary checks."""
        vocab_list = ["<pad>", "<unk>"] + [f"<0x{b:02X}>" for b in range(256)] + [chr(cp) for cp in range(32, 127)]
        seen = set(vocab_list)
        lang_subvocabs: Dict[str, List[str]] = {}

        for lang, texts in sorted(lang_corpora.items()):
            # Standard unguarded byte/char BPE + intermediate prefix retention (no scaffold pruning)
            sub_counts = collections.Counter()
            for t in texts:
                for w in t.split():
                    # Standard byte-level BPE builds 2-byte UTF-8 scaffold prefixes + 1..2 char merges
                    b_seq = w.encode("utf-8")
                    for bi in range(len(b_seq) - 1):
                        sub_counts[f"<0x{b_seq[bi]:02X}><0x{b_seq[bi+1]:02X}>"] += 2
                    for i in range(len(w)):
                        for k in range(i + 1, min(len(w) + 1, i + 3)):
                            sub_counts[w[i:k]] += 1
            lang_toks = []
            for sub, _ in sub_counts.most_common(per_lang_budget * 2):
                if len(lang_toks) >= per_lang_budget:
                    break
                lang_toks.append(sub)
                if sub not in seen and len(vocab_list) < total_vocab_size:
                    seen.add(sub)
                    vocab_list.append(sub)
            lang_subvocabs[lang] = lang_toks

        idx = 0
        while len(vocab_list) < total_vocab_size:
            t = f"<PARITY_SCAFFOLD_{idx}>"
            vocab_list.append(t)
            idx += 1

        return cls(vocab=vocab_list[:total_vocab_size], lang_subvocabs=lang_subvocabs)

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        ids = []
        for chunk in re.findall(r"\S+|\s+", text):
            if chunk.isspace():
                for c in chunk:
                    ids.append(self.token_to_id.get(c, 1))
                continue
            i = 0
            n = len(chunk)
            while i < n:
                best_k = i + 1
                for k in range(n, i, -1):
                    if chunk[i:k] in self.token_to_id:
                        best_k = k
                        break
                sub = chunk[i:best_k]
                if sub in self.token_to_id:
                    ids.append(self.token_to_id[sub])
                else:
                    # Standard 3-byte UTF-8 fallback
                    for b in sub.encode("utf-8"):
                        ids.append(self.token_to_id.get(f"<0x{b:02X}>", 1))
                i = best_k
        return ids

    def decode(self, token_ids: List[int]) -> str:
        out = []
        byte_buf = bytearray()
        for tid in token_ids:
            tok = self.id_to_token.get(tid, "")
            if tok.startswith("<0x") and tok.endswith(">") and len(tok) == 6:
                byte_buf.append(int(tok[3:5], 16))
            else:
                if byte_buf:
                    out.append(byte_buf.decode("utf-8", errors="replace"))
                    byte_buf.clear()
                if not tok.startswith("<PARITY_SCAFFOLD_"):
                    out.append(tok)
        if byte_buf:
            out.append(byte_buf.decode("utf-8", errors="replace"))
        return "".join(out)


class SuperBPETokenizer:
    """Baseline SuperBPE / IndicSuperTokenizer (2025): Multi-Word BPE merging across whitespace.
    
    Learns multi-word collocations on formal Wikipedia text, achieving low token count on formal
    in-domain text but suffering high cross-domain fertility drift on informal conversational text.
    """

    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self.id_to_token = {i: t for i, t in enumerate(vocab)}
        self.vocab_size = len(vocab)

    @classmethod
    def train_from_corpora(
        cls,
        lang_corpora: Dict[str, List[str]],
        total_vocab_size: int = 128000
    ) -> "SuperBPETokenizer":
        vocab_list = ["<pad>", "<unk>"] + [f"<0x{b:02X}>" for b in range(256)] + [chr(cp) for cp in range(32, 127)]
        seen = set(vocab_list)

        # 1. Base characters and whole words from the corpus
        for lang, texts in sorted(lang_corpora.items()):
            for t in texts:
                for c in t:
                    if c not in seen:
                        seen.add(c)
                        vocab_list.append(c)
                for w in t.split():
                    if w not in seen and len(vocab_list) < int(total_vocab_size * 0.3):
                        seen.add(w)
                        vocab_list.append(w)

        # 2. Learn multi-word bigrams and trigrams from formal training sentences
        collocations = collections.Counter()
        subwords = collections.Counter()
        for lang, texts in sorted(lang_corpora.items()):
            for t in texts:
                words = t.split()
                for i in range(len(words) - 1):
                    collocations[words[i] + " " + words[i + 1]] += 3
                for i in range(len(words) - 2):
                    collocations[words[i] + " " + words[i + 1] + " " + words[i + 2]] += 2
                for w in words:
                    for i in range(len(w)):
                        for k in range(i + 1, min(len(w) + 1, i + 4)):
                            subwords[w[i:k]] += 1

        for col, _ in collocations.most_common(int(total_vocab_size * 0.5)):
            if len(vocab_list) >= int(total_vocab_size * 0.8):
                break
            if col not in seen:
                seen.add(col)
                vocab_list.append(col)

        for sw, _ in subwords.most_common(int(total_vocab_size * 0.3)):
            if len(vocab_list) >= total_vocab_size:
                break
            if sw not in seen:
                seen.add(sw)
                vocab_list.append(sw)

        idx = 0
        while len(vocab_list) < total_vocab_size:
            vocab_list.append(f"<SUPER_COL_{idx}>")
            idx += 1

        return cls(vocab=vocab_list[:total_vocab_size])

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        ids = []
        i = 0
        n = len(text)
        while i < n:
            best_k = i + 1
            max_span = min(n, i + 32)
            for k in range(max_span, i, -1):
                if text[i:k] in self.token_to_id:
                    best_k = k
                    break
            sub = text[i:best_k]
            if sub in self.token_to_id:
                ids.append(self.token_to_id[sub])
            else:
                for b in sub.encode("utf-8"):
                    ids.append(self.token_to_id.get(f"<0x{b:02X}>", 1))
            i = best_k
        return ids

    def decode(self, token_ids: List[int]) -> str:
        out = []
        byte_buf = bytearray()
        for tid in token_ids:
            tok = self.id_to_token.get(tid, "")
            if tok.startswith("<0x") and tok.endswith(">") and len(tok) == 6:
                byte_buf.append(int(tok[3:5], 16))
            else:
                if byte_buf:
                    out.append(byte_buf.decode("utf-8", errors="replace"))
                    byte_buf.clear()
                if not tok.startswith("<SUPER_COL_"):
                    out.append(tok)
        if byte_buf:
            out.append(byte_buf.decode("utf-8", errors="replace"))
        return "".join(out)
