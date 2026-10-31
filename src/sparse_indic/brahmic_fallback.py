"""Module 6: Script-Relative 1-Byte Offset Fallback for Brahmic Unicode scripts."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# 128 Script-Relative Offset Token literals
BRAHMIC_OFFSET_TOKENS = [f"<BRAHMIC_0x{i:02X}>" for i in range(128)]

# Base code points for Brahmic scripts (each block spans exactly 128 code points)
BRAHMIC_BASES: Dict[str, int] = {
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


def get_active_block_base(lang_code: Optional[str]) -> Optional[int]:
    """Retrieve base Unicode code point for the active language."""
    if not lang_code:
        return None
    return BRAHMIC_BASES.get(lang_code)


def encode_brahmic_oov_char(c: str, lang_code: Optional[str] = None) -> Optional[str]:
    """Encode an out-of-vocabulary Brahmic character as a single Script-Relative Offset Token.
    
    Instead of 3 UTF-8 bytes (e.g. 0xE0, 0xA4, 0x95), returns a single '<BRAHMIC_0x{offset:02X}>'.
    """
    cp = ord(c)
    if not (0x0900 <= cp <= 0x0D7F):
        return None

    base = get_active_block_base(lang_code)
    if base is None:
        # Detect base automatically from code point range
        base = (cp // 0x80) * 0x80

    offset = cp - base
    if 0 <= offset < 128:
        return f"<BRAHMIC_0x{offset:02X}>"
    return None


def decode_brahmic_offset_token(token_str: str, lang_code: Optional[str] = None) -> Optional[str]:
    """Decode a '<BRAHMIC_0x{offset:02X}>' token back into its exact native Unicode character."""
    if not (token_str.startswith("<BRAHMIC_0x") and token_str.endswith(">") and len(token_str) == 14):
        return None

    try:
        hex_part = token_str[11:13]
        offset = int(hex_part, 16)
    except ValueError:
        return None

    base = get_active_block_base(lang_code)
    if base is None:
        base = 0x0900  # Default to Devanagari if unspecified

    return chr(base + offset)


class BrahmicFallbackHandler:
    """Manages 1-token Brahmic OOV fallback and detokenization."""

    def __init__(self, token_to_id: Dict[str, int], id_to_token: Dict[int, str]):
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token

    def is_brahmic_offset_token(self, token_id: int) -> bool:
        tok_str = self.id_to_token.get(token_id, "")
        return tok_str.startswith("<BRAHMIC_0x") and tok_str.endswith(">")

    def encode_char(self, c: str, lang_code: Optional[str]) -> Optional[int]:
        tok_str = encode_brahmic_oov_char(c, lang_code)
        if tok_str and tok_str in self.token_to_id:
            return self.token_to_id[tok_str]
        return None

    def decode_id(self, token_id: int, lang_code: Optional[str]) -> Optional[str]:
        tok_str = self.id_to_token.get(token_id, "")
        return decode_brahmic_offset_token(tok_str, lang_code)
