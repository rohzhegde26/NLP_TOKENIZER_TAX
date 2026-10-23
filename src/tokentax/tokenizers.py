"""One wrapper over tiktoken and Hugging Face tokenizers.

Contract (Day 1, do not change): get_tokenizer(name) returns an object with
encode(text) -> list[int], decode(ids) -> str, pieces(text) -> list[str].
Extra helpers (token_bytes, is_single_byte, has, id) are used by E1 and E4.
"""
from __future__ import annotations

from functools import lru_cache
import re

TOKENIZERS = ["cl100k", "o200k", "llama3", "qwen2.5", "gemma2", "xlmr", "sarvam1"]

HF_IDS = {
    "llama3": "meta-llama/Llama-3.2-1B",       # gated; same tokenizer as all Llama 3.x
    "qwen2.5": "Qwen/Qwen2.5-3B-Instruct",
    "gemma2": "google/gemma-2-2b-it",          # gated
    "xlmr": "FacebookAI/xlm-roberta-base",
    "sarvam1": "sarvamai/sarvam-1",            # may be gated
}
TIKTOKEN_IDS = {"cl100k": "cl100k_base", "o200k": "o200k_base"}
GATED = {"llama3", "gemma2", "sarvam1"}

_BYTE_FALLBACK = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte -> printable-char table (used by byte-level BPE vocabularies)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def _is_complete_utf8(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


class TiktokenTokenizer:
    def __init__(self, name: str):
        import tiktoken

        self.name = name
        self._enc = tiktoken.get_encoding(TIKTOKEN_IDS[name])

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, disallowed_special=())

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def token_bytes(self, tid: int) -> bytes:
        return self._enc.decode_single_token_bytes(tid)

    def pieces(self, text: str) -> list[str]:
        return [self.token_bytes(t).decode("utf-8", errors="replace") for t in self.encode(text)]

    def is_single_byte(self, tid: int) -> bool:
        return not _is_complete_utf8(self.token_bytes(tid))

    def has(self, b: bytes) -> bool:
        return b in self._enc._mergeable_ranks

    def id(self, b: bytes) -> int:
        return self._enc._mergeable_ranks[b]


class HFTokenizer:
    def __init__(self, name: str):
        from transformers import AutoTokenizer

        self.name = name
        self._tok = AutoTokenizer.from_pretrained(HF_IDS[name])
        # Byte-level BPE (GPT-2 style: Llama 3, Qwen) stores bytes as printable unicode.
        self._byte_encoder = _bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}
        self._byte_level = '"ByteLevel"' in self._tok.backend_tokenizer.to_str()
        self._vocab = self._tok.get_vocab()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, clean_up_tokenization_spaces=False)

    def token_bytes(self, tid: int) -> bytes:
        tok = self._tok.convert_ids_to_tokens(tid)
        m = _BYTE_FALLBACK.match(tok)
        if m:  # SentencePiece byte-fallback token
            return bytes([int(m.group(1), 16)])
        if self._byte_level:
            return bytes(self._byte_decoder[c] for c in tok)
        return tok.replace("\u2581", " ").encode("utf-8")

    def pieces(self, text: str) -> list[str]:
        return [self.token_bytes(t).decode("utf-8", errors="replace") for t in self.encode(text)]

    def is_single_byte(self, tid: int) -> bool:
        return not _is_complete_utf8(self.token_bytes(tid))

    def has(self, b: bytes) -> bool:
        if self._byte_level:
            token_str = "".join(self._byte_encoder[byte] for byte in b)
            return token_str in self._vocab
        if len(b) == 1:
            fallback = f"<0x{b[0]:02X}>"
            if fallback in self._vocab:
                return True
        s = b.decode("utf-8", errors="ignore")
        return bool(s and s in self._vocab)

    def id(self, b: bytes) -> int:
        if self._byte_level:
            token_str = "".join(self._byte_encoder[byte] for byte in b)
            if token_str in self._vocab:
                return self._vocab[token_str]
        elif len(b) == 1:
            fallback = f"<0x{b[0]:02X}>"
            if fallback in self._vocab:
                return self._vocab[fallback]
        else:
            s = b.decode("utf-8", errors="ignore")
            if s and s in self._vocab:
                return self._vocab[s]
        raise KeyError(f"bytes {b!r} not in vocabulary for {self.name}")


@lru_cache(maxsize=None)
def get_tokenizer(name: str):
    if name in TIKTOKEN_IDS:
        return TiktokenTokenizer(name)
    if name in HF_IDS:
        return HFTokenizer(name)
    raise ValueError(f"unknown tokenizer {name!r}; choose from {TOKENIZERS}")
