"""Akshara-Bounded Morphological Pre-Tokenizer and Suffix Splitter."""
from __future__ import annotations

import re
import sys
from typing import Dict, List, Optional, Set, Tuple

# Unicode code-point offset masks for Brahmic scripts (0x00 to 0x7F relative to block base)
# Signs: 0x01 (Chandrabindu), 0x02 (Anusvara), 0x03 (Visarga)
# Independent Vowels: 0x04 to 0x14
# Consonants: 0x15 to 0x39
# Nukta: 0x3C
# Matras (Dependent Vowels): 0x3E to 0x4C, 0x62 to 0x63
# Virama (Halant): 0x4D
# Additional signs: 0x50 to 0x61, 0x70 to 0x7F

BRAHMIC_BLOCK_BASES = {
    "hin_Deva": 0x0900,
    "mar_Deva": 0x0900,
    "npi_Deva": 0x0900,
    "ben_Beng": 0x0980,
    "asm_Beng": 0x0980,
    "pan_Guru": 0x0A00,
    "guj_Gujr": 0x0A80,
    "ory_Orya": 0x0B00,
    "tam_Taml": 0x0B80,
    "tel_Telu": 0x0C00,
    "kan_Knda": 0x0C80,
    "mal_Mlym": 0x0D00,
}

ZWJ = "\u200d"
ZWNJ = "\u200c"


def is_brahmic_char(c: str) -> bool:
    cp = ord(c)
    return 0x0900 <= cp <= 0x0D7F


def get_brahmic_char_role(c: str) -> str:
    """Return linguistic role of Brahmic character: 'consonant', 'indep_vowel', 'matra', 'halant', 'nukta', 'sign', or 'other'."""
    cp = ord(c)
    if not (0x0900 <= cp <= 0x0D7F):
        return "other"
    offset = cp & 0x7F

    if offset in (0x01, 0x02, 0x03):
        return "sign"
    elif 0x04 <= offset <= 0x14:
        return "indep_vowel"
    elif 0x15 <= offset <= 0x39:
        return "consonant"
    elif offset == 0x3C:
        return "nukta"
    elif (0x3E <= offset <= 0x4C) or (0x62 <= offset <= 0x63):
        return "matra"
    elif offset == 0x4D:
        return "halant"
    elif 0x58 <= offset <= 0x5F:
        return "consonant"
    return "sign"


def is_bare_dependent_char(c: str) -> bool:
    """True if character is a dependent matra, halant, or nukta that cannot start a valid token."""
    role = get_brahmic_char_role(c)
    return role in ("matra", "halant", "nukta")


class AksharaGraphemeSegmenter:
    """Splits Brahmic text into linguistic Aksharas (orthographic syllables) without breaking ligatures."""

    def __init__(self):
        # Akshara = (Consonant + Nukta? + Halant + [ZWJ/ZWNJ]?)* + Consonant + Nukta? + Dependent_Matra* + Sign?
        # Or: Independent_Vowel + Sign?
        pass

    def segment_word(self, word: str) -> List[str]:
        if not word:
            return []
        aksharas = []
        n = len(word)
        idx = 0

        while idx < n:
            c = word[idx]
            if not is_brahmic_char(c):
                # Non-Brahmic char (Latin, digit, punct): emit as single char
                aksharas.append(c)
                idx += 1
                continue

            role = get_brahmic_char_role(c)

            if role == "indep_vowel":
                curr = c
                idx += 1
                # Absorb following signs
                while idx < n and get_brahmic_char_role(word[idx]) == "sign":
                    curr += word[idx]
                    idx += 1
                aksharas.append(curr)

            elif role in ("consonant", "nukta", "sign", "matra", "halant"):
                # Grapheme cluster with consonants, halants, matras
                curr = c
                idx += 1
                # Consume conjuncts: (Nukta? + Halant + Joiner? + Consonant)*
                while idx < n:
                    next_c = word[idx]
                    next_role = get_brahmic_char_role(next_c)

                    if next_role == "nukta":
                        curr += next_c
                        idx += 1
                    elif next_role == "halant":
                        curr += next_c
                        idx += 1
                        # Check optional joiner
                        if idx < n and word[idx] in (ZWJ, ZWNJ):
                            curr += word[idx]
                            idx += 1
                        # If next is consonant, it's part of the same conjunct cluster
                        if idx < n and get_brahmic_char_role(word[idx]) == "consonant":
                            curr += word[idx]
                            idx += 1
                        else:
                            break
                    elif next_role == "matra":
                        curr += next_c
                        idx += 1
                    elif next_role == "sign":
                        curr += next_c
                        idx += 1
                        break
                    else:
                        break
                aksharas.append(curr)
            else:
                aksharas.append(c)
                idx += 1

        return aksharas


class MorphologicalSuffixSplitter:
    """Lightweight morpheme boundary detector for Dravidian agglutinated cases and Indo-Aryan inflections."""

    # High frequency inflectional suffixes and postpositional markers
    DRAVIDIAN_SUFFIXES = {
        "kan_Knda": [
            "ಗಳಲ್ಲಿ", "ಗಳಿಂದ", "ಗಳನ್ನು", "ಗಳಿಗೆ", "ದಲ್ಲಿ", "ನಲ್ಲಿ", "ಗೆ", "ಕ್ಕೆ",
            "ಅಲ್ಲಿ", "ಇಂದ", "ಅನ್ನು", "ಗಳು", "ದ", "ರ"
        ],
        "tam_Taml": [
            "களில்", "களை", "களுக்கு", "களின்", "இருந்து", "ந்து", "க்கு", "இல்", "ஐ", "கள்"
        ],
        "tel_Telu": [
            "నుండి", "లో", "తో", "కి", "కు", "లు", "ని", "ను"
        ],
        "mal_Mlym": [
            "കളിൽനിന്ന്", "ങ്ങളിൽനിന്ന്", "നിന്ന്", "ൽ", "ളെ", "കൾ", "ക്ക്", "ഇൽ"
        ],
        "hin_Deva": [
            "कर्ताओं", "वालों", "वाली", "वाले", "वाला", "ियों", "यों",
            "ाएं", "ों", "कर", "ते", "ती", "ता", "ने", "ना", "नी"
        ],
        "mar_Deva": [
            "मधील", "तून", "कडे", "वरून", "मुळे", "साठी", "च्या", "चे", "ची", "ना", "नी"
        ],
        "ben_Beng": [
            "গুলিতে", "গুলোর", "গুলো", "দের", "কে", "তে", "র", "এর"
        ]
    }

    def __init__(self):
        # Compile descending length regex patterns for each language
        self.patterns: Dict[str, List[str]] = {}
        for lang, sfx_list in self.DRAVIDIAN_SUFFIXES.items():
            # Sort longest suffix first
            sorted_sfx = sorted(sfx_list, key=len, reverse=True)
            self.patterns[lang] = sorted_sfx

    def split_word(self, word: str, lang: Optional[str] = None) -> List[str]:
        """Split agglutinated word into stem and suffix boundary while preserving akshara integrity."""
        if not lang or lang not in self.patterns or len(word) < 4:
            return [word]

        for sfx in self.patterns[lang]:
            if word.endswith(sfx) and len(word) > len(sfx) + 1:
                stem = word[:-len(sfx)]
                # Ensure the split does not orphan a bare dependent matra or halant at suffix start
                if is_bare_dependent_char(sfx[0]):
                    # Attach the initial matra/halant to the stem
                    stem = stem + sfx[0]
                    clean_sfx = sfx[1:]
                    if clean_sfx:
                        return [stem, clean_sfx]
                else:
                    return [stem, sfx]

        return [word]


class AksharaBoundedPreTokenizer:
    """Pre-tokenizer combining Akshara boundary protection, morphological splitting, and single-word cap."""

    def __init__(self, enforce_single_word: bool = True):
        self.segmenter = AksharaGraphemeSegmenter()
        self.suffix_splitter = MorphologicalSuffixSplitter()
        self.enforce_single_word = enforce_single_word

    def pretokenize(self, text: str, lang: Optional[str] = None) -> List[str]:
        """Split input text into safe pre-token chunks where subwords may be constructed."""
        if not text:
            return []

        # Enforce single-word cap via whitespace & punctuation splitting (preserving Brahmic marks)
        tokens = re.findall(r"[\u0900-\u0D7F\w]+|[^\s\w]", text, re.UNICODE)
        output_chunks = []

        for token in tokens:
            if re.match(r"[^\w\s]", token):
                output_chunks.append(token)
                continue

            # Morphological pre-splitting on Dravidian / agglutinated words
            morps = self.suffix_splitter.split_word(token, lang=lang)
            for m in morps:
                # Segment into akshara clusters to ensure subwords never breach cluster boundaries
                aksharas = self.segmenter.segment_word(m)
                output_chunks.extend(aksharas)

        return output_chunks

    def is_valid_subword(self, subword: str) -> bool:
        """Verify that a subword satisfies the Akshara-Boundary Guard (never begins with bare matra/halant)."""
        if not subword:
            return False
        stripped = subword.lstrip(" \t\n\r")
        if not stripped:
            return True
        first_char = stripped[0]
        if is_bare_dependent_char(first_char):
            return False
        return True
