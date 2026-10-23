"""Contract, metric and schema tests (no network)."""
import pandas as pd
import pytest

from tokentax.metrics import summarize_e1
from tokentax.schemas import COLUMNS, check
from tokentax.stats import bootstrap_ci, holm, mcnemar
from tokentax.tokenizers import TOKENIZERS, get_tokenizer


def test_tokenizer_names():
    assert TOKENIZERS == ["cl100k", "o200k", "llama3", "qwen2.5", "gemma2", "xlmr", "sarvam1"]


@pytest.mark.parametrize("name", ["cl100k", "o200k"])
def test_tiktoken_roundtrip_and_single_byte(name):
    tok = get_tokenizer(name)
    s = "ಕನ್ನಡ ಒಂದು ಸುಂದರ ಭಾಷೆ. Hello world."
    ids = tok.encode(s)
    assert tok.decode(ids) == s
    assert "".join(tok.pieces("hello world")) == "hello world"
    assert not any(tok.is_single_byte(i) for i in tok.encode("hello world"))


@pytest.mark.parametrize("name", list(COLUMNS))
def test_fake_csvs_match_contract(name):
    df = pd.read_csv(f"results/fake/{name}.csv")
    check(df, name)


def test_english_tax_is_one():
    df = pd.read_csv("results/fake/e1_intrinsic.csv")
    s = summarize_e1(df)
    assert (s[s.lang == "eng_Latn"].tax_vs_eng == 1.0).all()


def test_stats():
    lo, hi = bootstrap_ci([1, 0] * 50)
    assert lo < 0.5 < hi
    assert mcnemar([1, 1, 0], [1, 1, 0]) == 1.0
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


@pytest.mark.parametrize("name", ["cl100k", "o200k"])
def test_has_and_id_for_resegmentation(name):
    tok = get_tokenizer(name)
    assert all(tok.has(bytes([b])) for b in range(256))  # byte-level vocab: every single byte exists
    for tid in tok.encode("മലയാളം hello"):
        assert tok.id(tok.token_bytes(tid)) == tid
    assert tok.id(b"\xff\xfe\xfd\xfc this is not one token") is None
