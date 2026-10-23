"""Language codes (FLORES codes everywhere, never 'kn' or 'Kannada')."""

ENGLISH = "eng_Latn"

# E1-E2 tokenizer audit: English + 12 Indian languages.
AUDIT_LANGS = [
    "eng_Latn", "hin_Deva", "mar_Deva", "ben_Beng", "asm_Beng", "guj_Gujr",
    "pan_Guru", "ory_Orya", "urd_Arab", "tam_Taml", "tel_Telu", "kan_Knda",
    "mal_Mlym",
]

# E3-E4 GPU experiments.
GPU_LANGS = ["eng_Latn", "hin_Deva", "ben_Beng", "tam_Taml", "tel_Telu", "kan_Knda", "mal_Mlym"]

LANG_NAMES = {
    "eng_Latn": "English", "hin_Deva": "Hindi", "mar_Deva": "Marathi",
    "ben_Beng": "Bengali", "asm_Beng": "Assamese", "guj_Gujr": "Gujarati",
    "pan_Guru": "Punjabi", "ory_Orya": "Odia", "urd_Arab": "Urdu",
    "tam_Taml": "Tamil", "tel_Telu": "Telugu", "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
}

FAMILY = {c: "Dravidian" for c in ["tam_Taml", "tel_Telu", "kan_Knda", "mal_Mlym"]}
FAMILY.update({c: "Indo-Aryan" for c in AUDIT_LANGS if c not in FAMILY and c != ENGLISH})
FAMILY[ENGLISH] = "Germanic"
