"""Automated verification suite for the Comparative Tokenizer Interactive Demo."""
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo"
DEMO_DATA_JSON = DEMO_DIR / "data" / "demo_data.json"
DEMO_DATA_JS = DEMO_DIR / "data" / "demo_data.js"
DEMO_HTML = DEMO_DIR / "index.html"
FIGURES_DIR = DEMO_DIR / "figures"


def test_demo_files_exist():
    """Verify demo files and directories exist."""
    assert DEMO_HTML.exists(), "demo/index.html must exist"
    assert DEMO_DATA_JSON.exists(), "demo/data/demo_data.json must exist"
    assert DEMO_DATA_JS.exists(), "demo/data/demo_data.js must exist"
    assert FIGURES_DIR.exists(), "demo/figures directory must exist"


def test_demo_data_integrity():
    """Verify demo payload contains all required probes, figures, and models."""
    with open(DEMO_DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Probes
    probes = data.get("probes", [])
    assert len(probes) >= 16, f"Expected >= 16 probes, found {len(probes)}"

    required_tokenizers = [
        "sparse_indic", "cl100k_base", "o200k_base", "qwen2_5", "qwen3_5",
        "gemma_2", "gemma_4", "lfm_3", "xlm_roberta", "mbert", "indicbert_v2",
        "parity_bpe", "super_bpe"
    ]

    for p in probes:
        t_dict = p.get("tokenizers", {})
        for req_t in required_tokenizers:
            assert req_t in t_dict, f"Tokenizer '{req_t}' missing in probe '{p['probe_info']['id']}'"
            t_entry = t_dict[req_t]
            assert "tokens" in t_entry
            assert "token_count" in t_entry
            assert t_entry["token_count"] > 0

    # 2. Scorecard
    scorecard = data.get("scorecard", [])
    assert len(scorecard) == 13, f"Expected 13 rows in scorecard, got {len(scorecard)}"

    # 3. Figures & Paper Graphs
    figures = data.get("figures", [])
    assert len(figures) >= 8, f"Expected >= 8 figures, got {len(figures)}"
    for fig in figures:
        fn = fig["filename"]
        img_path = FIGURES_DIR / fn
        assert img_path.exists(), f"Figure file '{fn}' does not exist in {FIGURES_DIR}"
        assert img_path.stat().st_size > 10000, f"Figure '{fn}' is corrupted or empty"


def test_demo_html_components():
    """Verify index.html contains essential DOM elements and functions."""
    html_text = DEMO_HTML.read_text(encoding="utf-8")
    assert 'id="main-text-input"' in html_text
    assert 'id="sparse-token-stream"' in html_text
    assert 'id="rival-token-stream"' in html_text
    assert 'id="scorecard-tbody"' in html_text
    assert 'id="presets-container"' in html_text
    assert 'data/demo_data.js' in html_text
