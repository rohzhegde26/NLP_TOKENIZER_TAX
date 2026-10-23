"""Test resegmentation roundtrip and invariance across languages (Lane C, D1)."""
from __future__ import annotations

import pytest
from tokentax.eval.resegment import match_fragment, resegment, to_single_bytes
from tokentax.tokenizers import get_tokenizer

SAMPLE_TEXTS = {
    "eng_Latn": [
        "The quick brown fox jumps over the lazy dog.",
        "Tokenization parity is essential for fair multilingual language models.",
        "Context loss and token inflation disproportionately hurt low-resource languages.",
    ],
    "hin_Deva": [
        "भारतीय भाषाओं के लिए टोकनाइज़र टैक्स एक गंभीर समस्या है।",
        "कम संसाधन वाली भाषाओं को अधिक टोकन की आवश्यकता होती है।",
    ],
    "ben_Beng": [
        "টোকেনাইজেশন প্রক্রিয়ায় ভারতীয় ভাষাগুলো পিছিয়ে রয়েছে।",
        "কম্পিউটার বিজ্ঞানে প্রাকৃতিক ভাষা প্রক্রিয়াকরণ একটি গুরুত্বপূর্ণ ক্ষেত্র।",
    ],
    "tam_Taml": [
        "இந்திய மொழிகளில் டோக்கனாக்க செயல்முறை கூடுதல் செலவை உருவாக்குகிறது.",
        "தமிழ் மொழி நீண்ட வரலாற்றையும் செழுமையான இலக்கியத்தையும் கொண்டுள்ளது.",
    ],
    "tel_Telu": [
        "భారతీయ భాషలకు టోకనైజర్ ఖర్చు చాలా ఎక్కువగా ఉంది.",
        "తెలుగు భాష ద్రావిడ భాషా కుటుంబానికి చెందిన ఒక ప్రముఖ భాష.",
    ],
    "kan_Knda": [
        "ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ ಟೋಕನ್ ವಿಭಜನೆಯು ಹೆಚ್ಚು ತುಣುಕುಗಳನ್ನು ಉಂಟುಮಾಡುತ್ತದೆ.",
        "ನೈಸರ್ಗಿಕ ಭಾಷಾ ಸಂಸ್ಕರಣೆಯಲ್ಲಿ ಇದು ಮಹತ್ವದ ಸವಾಲಾಗಿದೆ.",
    ],
    "mal_Mlym": [
        "മലയാളം ഭാഷയിൽ ടോക്കണൈസേഷൻ കൂടുതൽ ബൈറ്റ് ലെവൽ വിഭജനങ്ങൾക്ക് കാരണമാകുന്നു.",
        "ഇത് വലിയ തോതിലുള്ള സാന്ദർഭിക നഷ്ടത്തിന് വഴിവെക്കുന്നു.",
    ],
}


@pytest.mark.parametrize("p", [0.0, 0.1, 0.3, 0.5])
def test_resegment_roundtrip_qwen(p: float):
    tok = get_tokenizer("qwen2.5")
    for lang, texts in SAMPLE_TEXTS.items():
        for text in texts:
            ids = tok.encode(text)
            re_ids = resegment(tok, ids, p=p, seed=42)
            assert tok.decode(re_ids) == text
            if p == 0:
                assert re_ids == ids
            else:
                assert len(re_ids) >= len(ids)


@pytest.mark.parametrize("tok_name", ["cl100k", "o200k"])
def test_resegment_roundtrip_tiktoken(tok_name: str):
    tok = get_tokenizer(tok_name)
    for lang, texts in SAMPLE_TEXTS.items():
        for text in texts:
            ids = tok.encode(text)
            re_ids = resegment(tok, ids, p=0.3, seed=42)
            assert tok.decode(re_ids) == text
            assert len(re_ids) >= len(ids)


def test_to_single_bytes_roundtrip():
    tok = get_tokenizer("qwen2.5")
    for lang, texts in SAMPLE_TEXTS.items():
        for text in texts:
            ids = tok.encode(text)
            sb_ids = to_single_bytes(tok, ids)
            assert tok.decode(sb_ids) == text
            assert len(sb_ids) >= len(ids)


def test_match_fragment_properties():
    tok = get_tokenizer("qwen2.5")
    en_text = "The tokenizer tax reduces available context window for non-English languages."
    en_ids = tok.encode(en_text)

    # Match target length = 2x English
    target_len = len(en_ids) * 2
    mf_ids, capped = match_fragment(tok, en_ids, target_len=target_len, rng=42)
    assert tok.decode(mf_ids) == en_text
    assert len(mf_ids) == target_len or capped
