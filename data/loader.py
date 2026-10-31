"""Data loader for FLORES-200, Belebele, and Code-Mix probe suites."""
from __future__ import annotations

import json
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CONFIG_PATH = REPO_ROOT / "configs" / "languages.yaml"

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


def load_languages_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["languages"]


def ensure_flores_archive() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = CACHE_DIR / "flores200_dataset.tar.gz"
    if not tar_path.exists() or tar_path.stat().st_size < 1000000:
        print(f"Downloading FLORES-200 from {FLORES_URL} ...")
        urllib.request.urlretrieve(FLORES_URL, tar_path)
    return tar_path


def get_flores_sentences(lang_code: str, split: str = "devtest") -> List[str]:
    """Retrieve aligned FLORES-200 sentences for a given BCP-47 language code.
    
    Returns exactly 1,012 sentences for devtest and 997 for dev.
    """
    tar_path = ensure_flores_archive()
    cached_text_file = CACHE_DIR / f"flores_{split}_{lang_code}.txt"
    if cached_text_file.exists():
        with open(cached_text_file, "r", encoding="utf-8") as f:
            return [line.rstrip("\r\n") for line in f]

    target_entry = f"./flores200_dataset/{split}/{lang_code}.{split}"
    alt_entry = f"flores200_dataset/{split}/{lang_code}.{split}"

    with tarfile.open(tar_path, "r:gz") as tar:
        member = None
        for candidate in (target_entry, alt_entry):
            try:
                member = tar.getmember(candidate)
                break
            except KeyError:
                continue
        if member is None:
            raise FileNotFoundError(f"Language split not found in FLORES-200: {target_entry}")
        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Could not extract {target_entry}")
        lines = [line.decode("utf-8").rstrip("\r\n") for line in extracted]

    with open(cached_text_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")

    return lines


def get_belebele_data(lang_code: str) -> List[Dict[str, Any]]:
    """Retrieve Belebele multiple choice evaluation set for a given language.
    
    Caches locally in data/cache/belebele_{lang}.json for fast offline reproducibility.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_file = CACHE_DIR / f"belebele_{lang_code}.json"
    if cached_file.exists():
        with open(cached_file, "r", encoding="utf-8") as f:
            return json.load(f)

    from datasets import load_dataset
    print(f"Loading Belebele for {lang_code} from Hugging Face...")
    ds = load_dataset("facebook/belebele", lang_code, split="test")
    records = []
    for row in ds:
        records.append({
            "link": row.get("link", ""),
            "question_number": row.get("question_number", 0),
            "flores_passage": row.get("flores_passage", ""),
            "question": row.get("question", ""),
            "mc_answer1": row.get("mc_answer1", ""),
            "mc_answer2": row.get("mc_answer2", ""),
            "mc_answer3": row.get("mc_answer3", ""),
            "mc_answer4": row.get("mc_answer4", ""),
            "correct_answer_num": int(row.get("correct_answer_num", 1)),
            "dialect": row.get("dialect", ""),
            "language": lang_code,
        })

    with open(cached_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return records


def get_code_mix_probes() -> Dict[str, Any]:
    probes_path = DATA_DIR / "code_mix_probes.json"
    with open(probes_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    langs = list(load_languages_config().keys())
    print(f"Loaded config for {len(langs)} languages: {langs}")
    en_sentences = get_flores_sentences("eng_Latn", "devtest")
    hi_sentences = get_flores_sentences("hin_Deva", "devtest")
    print(f"FLORES-200 eng_Latn devtest sentences: {len(en_sentences)}")
    print(f"FLORES-200 hin_Deva devtest sentences: {len(hi_sentences)}")
    assert len(en_sentences) == 1012
    assert len(hi_sentences) == 1012
    probes = get_code_mix_probes()
    print(f"Code-mix probes: {len(probes['code_mix_pairs'])} pairs, {len(probes['conversational_indic'])} conversational, {len(probes['morphological_dravidian_test'])} morphological items.")
    print("Data loader test passed successfully.")
