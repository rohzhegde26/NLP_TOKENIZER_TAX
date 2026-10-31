"""Module 1: Two-Tier Unicode/Keyword Prompt Router & Sentinel Switch Tokens."""
from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Set, Tuple

# Mapping of Brahmic Unicode block ranges to script codes
SCRIPT_UNICODE_RANGES = {
    "Deva": (0x0900, 0x097F),
    "Beng": (0x0980, 0x09FF),
    "Guru": (0x0A00, 0x0A7F),
    "Gujr": (0x0A80, 0x0AFF),
    "Orya": (0x0B00, 0x0B7F),
    "Taml": (0x0B80, 0x0BFF),
    "Telu": (0x0C00, 0x0C7F),
    "Knda": (0x0C80, 0x0CFF),
    "Mlym": (0x0D00, 0x0D7F),
}

# Primary language corresponding to script
SCRIPT_TO_DEFAULT_LANG = {
    "Deva": "hin_Deva",
    "Beng": "ben_Beng",
    "Guru": "pan_Guru",
    "Gujr": "guj_Gujr",
    "Orya": "ory_Orya",
    "Taml": "tam_Taml",
    "Telu": "tel_Telu",
    "Knda": "kan_Knda",
    "Mlym": "mal_Mlym",
}

# Fast keyword patterns for Romanized code-mixed text
LANGUAGE_KEYWORDS: Dict[str, List[str]] = {
    "hin_Deva": ["hindi", "hinglish", "namaste", "bhai", "kya", "aur", "achha", "bahut"],
    "tam_Taml": ["tamil", "tanglish", "vanakkam", "nanri", "machi", "irukku", "enna"],
    "kan_Knda": ["kannada", "kanglish", "namaskara", "chennagide", "maga", "yochis"],
    "tel_Telu": ["telugu", "tenglish", "namaskaram", "bagundi", "enti", "cheppandi"],
    "ben_Beng": ["bengali", "bangla", "banglish", "nomoshkar", "bhalo", "khobor"],
    "mar_Deva": ["marathi", "marathiglish", "namaskar", "chalalay", "kasa"],
    "guj_Gujr": ["gujarati", "gujlish", "kem", "chho", "majama"],
    "pan_Guru": ["punjabi", "sat", "sri", "akaal", "veere"],
    "mal_Mlym": ["malayalam", "manglish", "namaskaram", "aliya", "vishesham"],
    "ory_Orya": ["odia", "oriya", "namaskar", "kemiti", "achhanti"],
    "npi_Deva": ["nepali", "namaste", "sanchai"],
    "asm_Beng": ["assamese", "asamiya", "nomoskar"],
}


class TwoTierActivationRouter:
    """Ultra-fast prompt router assembling active vocabulary indices in < 1 ms."""

    def __init__(
        self,
        core_vocab_size: int,
        bank_offsets: Dict[str, int],
        bank_size: int = 8192
    ):
        self.core_vocab_size = core_vocab_size
        self.bank_offsets = bank_offsets
        self.bank_size = bank_size
        self.core_indices = list(range(core_vocab_size))

        # Precompute bank slices for instantaneous assembly
        self.bank_slices = {
            lang: list(range(offset, offset + bank_size))
            for lang, offset in bank_offsets.items()
        }

        # Precompute keyword sets for O(1) set intersection
        self.lang_keyword_sets = {
            lang: set(kws) for lang, kws in LANGUAGE_KEYWORDS.items()
        }

        # Sentinel tokens map
        self.sentinel_to_lang = {
            f"<SWITCH:{lang}>": lang for lang in bank_offsets.keys()
        }

    def detect_languages(self, text: str) -> Set[str]:
        """Scan prompt for Unicode scripts and Romanized keywords in < 0.2 ms."""
        detected = set()
        if not text:
            return detected

        # Tier 1: Fast Unicode block scanner
        for char in text:
            cp = ord(char)
            if 0x0900 <= cp <= 0x0D7F:
                for script, (start, end) in SCRIPT_UNICODE_RANGES.items():
                    if start <= cp <= end:
                        lang = SCRIPT_TO_DEFAULT_LANG[script]
                        detected.add(lang)
                        break

        # Tier 2: Set-based keyword scanner for Romanized / code-mixed prompts
        text_words = set(re.findall(r"\w+", text.lower()))
        for lang, kw_set in self.lang_keyword_sets.items():
            if lang not in detected and not kw_set.isdisjoint(text_words):
                detected.add(lang)

        return detected

    def assemble_active_indices(self, detected_langs: Set[str]) -> List[int]:
        """Assemble active indices: I_core union sum_{k in detected} I_lang^(k)."""
        active = list(self.core_indices)
        for lang in detected_langs:
            if lang in self.bank_slices:
                active.extend(self.bank_slices[lang])
        return active

    def route_prompt(self, text: str) -> Tuple[List[int], Set[str], float]:
        """Route prompt text to active vocabulary indices and measure latency."""
        t0 = time.perf_counter()
        detected = self.detect_languages(text)
        active_indices = self.assemble_active_indices(detected)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return active_indices, detected, elapsed_ms

    def handle_sentinel_token(self, token_str: str, current_active_indices: List[int]) -> List[int]:
        """If generator emits a sentinel token <SWITCH:lang>, activate that bank."""
        if token_str in self.sentinel_to_lang:
            target_lang = self.sentinel_to_lang[token_str]
            if target_lang in self.bank_offsets:
                offset = self.bank_offsets[target_lang]
                new_slice = set(range(offset, offset + self.bank_size))
                curr_set = set(current_active_indices)
                curr_set.update(new_slice)
                return sorted(list(curr_set))
        return current_active_indices
