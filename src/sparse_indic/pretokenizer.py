"""Akshara-Bounded Morphological Pre-Tokenizer, Unsupervised Suffix Discovery, and Dual Code-Mix Routing."""
from __future__ import annotations

import collections
import math
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

# Phonetic Brahmic-Offset-to-Roman mappings for Dual Code-Mix Core Vocabulary Construction
CONSONANT_ROMAN = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
    0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
}
TAMIL_INTERVOCALIC_VOICED = {0x15: "g", 0x1A: "s", 0x1F: "d", 0x24: "dh", 0x2A: "b"}
VOWEL_ROMAN = {
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo",
    0x0B: "ri", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au",
}
MATRA_ROMAN = {
    0x3E: "aa", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo",
    0x43: "ri", 0x46: "e", 0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au",
}
SIGN_ROMAN = {
    0x01: "n", 0x02: "n", 0x03: "h",
    0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l",  # Malayalam chillu consonants
}


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


def brahmic_to_roman_variants(word: str, lang_code: str = "hin_Deva") -> Set[str]:
    """Generate canonical and colloquial Romanized forms of a Brahmic word via Unicode offset isomorphism."""
    variants: Set[str] = set()
    inherent_vowels = ["a", "o"] if lang_code in ("ben_Beng", "asm_Beng", "ory_Orya") else ["a"]
    voice_opts = [False, True] if lang_code == "tam_Taml" else [False]
    schwa_opts = [True, False] if lang_code in ("hin_Deva", "mar_Deva", "guj_Gujr", "pan_Guru", "ben_Beng") else [False]
    dental_th_opts = [False, True] if lang_code in ("tam_Taml", "kan_Knda", "tel_Telu", "mal_Mlym") else [False]

    n = len(word)
    for inherent in inherent_vowels:
        for voice in voice_opts:
            for schwa_del in schwa_opts:
                for dental_th in dental_th_opts:
                    out = []
                    for i, c in enumerate(word):
                        cp = ord(c)
                        if not (0x0900 <= cp <= 0x0D7F):
                            continue
                        off = cp & 0x7F
                        if off in CONSONANT_ROMAN:
                            is_gem = (
                                (i > 0 and (ord(word[i - 1]) & 0x7F) == 0x4D)
                                or (i + 1 < n and (ord(word[i + 1]) & 0x7F) == 0x4D)
                            )
                            if voice and i > 0 and not is_gem:
                                cons = TAMIL_INTERVOCALIC_VOICED.get(off, CONSONANT_ROMAN[off])
                            elif dental_th and off == 0x24:
                                cons = "th"
                            elif lang_code in ("ben_Beng", "asm_Beng") and off == 0x38:
                                cons = "sh"
                            else:
                                cons = CONSONANT_ROMAN[off]
                            out.append(cons)
                            if i + 1 < n:
                                next_off = ord(word[i + 1]) & 0x7F
                                if next_off not in MATRA_ROMAN and next_off != 0x4D:
                                    out.append(inherent)
                            elif not schwa_del:
                                out.append(inherent)
                        elif off in VOWEL_ROMAN:
                            out.append(VOWEL_ROMAN[off])
                        elif off in MATRA_ROMAN:
                            out.append(MATRA_ROMAN[off])
                        elif off in SIGN_ROMAN:
                            out.append(SIGN_ROMAN[off])
                    s = "".join(out)
                    if len(s) >= 2:
                        variants.add(s)
                        short_s = s.replace("aa", "a").replace("ee", "i").replace("oo", "u")
                        variants.add(short_s)

                        # Colloquial texting phonology adaptations (degemination, diphthongs, and script assimilation)
                        d1 = s.replace("chch", "ch").replace("thth", "th").replace("shsh", "sh").replace("jhjh", "jh")
                        variants.add(d1)
                        variants.add(d1.replace("tth", "th").replace("ddh", "d"))
                        variants.add(s.replace("nb", "mb").replace("np", "mp"))
                        variants.add(d1.replace("nb", "mb").replace("np", "mp"))
                        if lang_code == "hin_Deva":
                            variants.add(s.replace("men", "mein").replace("lie", "liye").replace("aen", "ayein").replace("en", "ein"))
                        if lang_code in ("ben_Beng", "asm_Beng"):
                            if s.startswith("y"):
                                variants.add("j" + s[1:])
                            variants.add(s.replace("aoya", "aowa").replace("aoy", "aw"))
                            variants.add(s.replace("on", "ong").replace("an", "ang"))
                            variants.add(s.replace("chhi", "chi"))
                        if lang_code == "mal_Mlym":
                            variants.add(s.replace("ny", "nj"))
                        if lang_code == "tel_Telu" and s.endswith("n"):
                            variants.add(s[:-1] + "m")
                        if lang_code == "tam_Taml":
                            variants.add(s.replace("nra", "ndra").replace("rra", "tra").replace("rru", "tru"))
                        if lang_code == "kan_Knda":
                            variants.add(s.replace("vatt", "vat").replace("iddene", "iddeene").replace("thth", "th"))
    return variants


class AksharaGraphemeSegmenter:
    """Splits Brahmic text into linguistic Aksharas (orthographic syllables) without breaking ligatures."""

    def segment_word(self, word: str) -> List[str]:
        if not word:
            return []
        aksharas = []
        n = len(word)
        idx = 0

        while idx < n:
            c = word[idx]
            if not is_brahmic_char(c):
                aksharas.append(c)
                idx += 1
                continue

            role = get_brahmic_char_role(c)

            if role == "indep_vowel":
                curr = c
                idx += 1
                while idx < n and get_brahmic_char_role(word[idx]) == "sign":
                    curr += word[idx]
                    idx += 1
                aksharas.append(curr)

            elif role in ("consonant", "nukta", "sign", "matra", "halant"):
                curr = c
                idx += 1
                while idx < n:
                    next_c = word[idx]
                    next_role = get_brahmic_char_role(next_c)

                    if next_role == "nukta":
                        curr += next_c
                        idx += 1
                    elif next_role == "halant":
                        curr += next_c
                        idx += 1
                        if idx < n and word[idx] in (ZWJ, ZWNJ):
                            curr += word[idx]
                            idx += 1
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
    """Morpheme boundary detector combining Dravidian/Indo-Aryan inflectional paradigms with unsupervised suffix entropy."""

    # Core grammatical case markers, plural suffixes, and postpositional enclitics across scripts
    INFLECTIONAL_PARADIGMS = {
        "kan_Knda": [
            "ಗಳಲ್ಲಿ", "ಗಳಿಂದ", "ಗಳನ್ನು", "ಗಳಿಗೆ", "ದಲ್ಲಿ", "ನಲ್ಲಿ",
            "ಗೆ", "ಕ್ಕೆ", "ಅಲ್ಲಿ", "ಇಂದ", "ಅನ್ನು", "ಗಳು"
        ],
        "tam_Taml": [
            "களில்", "களை", "களுக்கு", "களின்", "இருந்து", "ந்து", "க்கு", "இல்", "கள்"
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
        ],
        "asm_Beng": [
            "বোৰৰ", "বোৰক", "বোৰ", "সকল", "লৈ", "ত", "ক", "ৰ"
        ],
        "guj_Gujr": [
            "માંથી", "માં", "થી", "ને", "નું", "ના", "ની", "નો"
        ],
        "pan_Guru": [
            "ੀਆਂ", "ਿਆਂ", "ਾਂ", "ਵਾਂ", "ਈ", "ਿਆ"
        ],
        "ory_Orya": [
            "ଗୁଡ଼ିକର", "ଗୁଡ଼ିକ", "মানଙ୍କର", "মানଙ୍କୁ", "মানେ", "ରେ", "କୁ", "ରୁ"
        ],
        "npi_Deva": [
            "हरूलाई", "हरूको", "हरूमा", "हरू", "लाई", "बाट", "देखि", "मा", "को", "का", "की", "ले"
        ],
    }

    def __init__(self, learned_suffixes: Optional[Dict[str, List[str]]] = None):
        self.patterns: Dict[str, List[str]] = {}
        for lang, sfx_list in self.INFLECTIONAL_PARADIGMS.items():
            combined = set(sfx_list)
            if learned_suffixes and lang in learned_suffixes:
                combined.update(learned_suffixes[lang])
            self.patterns[lang] = sorted(combined, key=len, reverse=True)

    def learn_from_corpus(self, texts: List[str], lang: str, top_k: int = 25) -> List[str]:
        """Discover high-branching-entropy suffixes from a monolingual training corpus."""
        words = set()
        for t in texts:
            for w in re.findall(r"[\u0900-\u0D7F]+", t):
                if len(w) >= 5:
                    words.add(w)

        # Count how many distinct stems precede each candidate suffix (2..6 chars)
        sfx_to_stems: Dict[str, Set[str]] = collections.defaultdict(set)
        for w in words:
            for k in range(2, min(7, len(w) - 2)):
                stem = w[:-k]
                sfx = w[-k:]
                if not is_bare_dependent_char(sfx[0]):
                    sfx_to_stems[sfx].add(stem[-1])

        # Filter suffixes that attach to diverse stem endings (branching diversity >= 5) and high stem count
        scored = []
        for sfx, end_chars in sfx_to_stems.items():
            if len(end_chars) >= 5:
                scored.append((len(end_chars) * math.log(1 + len(sfx)), sfx))

        scored.sort(reverse=True)
        discovered = [sfx for _, sfx in scored[:top_k]]
        existing = set(self.patterns.get(lang, []))
        existing.update(discovered)
        self.patterns[lang] = sorted(existing, key=len, reverse=True)
        return discovered

    def split_word(self, word: str, lang: Optional[str] = None) -> List[str]:
        """Split agglutinated word into stem and suffix boundary while preserving akshara integrity."""
        if not lang or lang not in self.patterns or len(word) < 4:
            return [word]

        for sfx in self.patterns[lang]:
            if word.endswith(sfx) and len(word) > len(sfx) + 1:
                stem = word[: -len(sfx)]
                # Ensure the split does not orphan a bare dependent matra or halant at suffix start
                if is_bare_dependent_char(sfx[0]):
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

        tokens = re.findall(r"[\u0900-\u0D7F\w]+|[^\s\w]", text, re.UNICODE)
        output_chunks = []

        for token in tokens:
            if re.match(r"[^\w\s]", token):
                output_chunks.append(token)
                continue

            morps = self.suffix_splitter.split_word(token, lang=lang)
            for m in morps:
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
