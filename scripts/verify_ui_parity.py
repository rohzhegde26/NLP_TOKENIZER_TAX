"""Automated Screenshot Capture and Visual Parity Verification Suite.

Uses Playwright Chrome to take high-resolution screenshots of all three demo styles:
1. Style 1: Modern Dark Glassmorphism (index.html)
2. Style 2: Clean Editorial Academic (academic.html)
3. Style 4: Modern Bento-Grid Dashboard (bento.html)
"""
from __future__ import annotations

import os
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS_DIR = REPO_ROOT / "demo" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

MOCKUPS_DIR = REPO_ROOT / "demo" / "figures"


def capture_all_styles() -> None:
    print("=" * 70)
    print("Capturing Screenshots across all 3 Demo Styles via Playwright Chrome...")
    print("=" * 70)

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        chrome_path = None

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chrome_path, headless=True)

        pages_to_capture = [
            ("index.html", "screenshot_style1_glassmorphism.png", "Style 1: Dark Glassmorphism"),
            ("academic.html", "screenshot_style2_academic.png", "Style 2: Editorial Academic"),
            ("bento.html", "screenshot_style4_bento.png", "Style 4: Bento-Grid Dashboard"),
        ]

        for html_file, out_name, label in pages_to_capture:
            page = browser.new_page(viewport={"width": 1600, "height": 1050})
            url = f"http://127.0.0.1:8080/{html_file}"
            print(f"Loading {label} from {url} ...")
            page.goto(url)
            page.wait_for_timeout(1500) # Ensure JS data is fully rendered

            out_path = SCREENSHOTS_DIR / out_name
            page.screenshot(path=str(out_path))
            print(f"  -> Captured viewport screenshot: {out_path} ({out_path.stat().st_size:,} bytes)")

            # Also capture full page
            full_out = SCREENSHOTS_DIR / out_name.replace(".png", "_fullpage.png")
            page.screenshot(path=str(full_out), full_page=True)
            print(f"  -> Captured fullpage screenshot: {full_out} ({full_out.stat().st_size:,} bytes)")

            page.close()

        browser.close()

    print("\nAll screenshots successfully captured!")


def create_side_by_side_parity_exhibits() -> None:
    """Generate side-by-side comparison images contrasting AI mockup vs real HTML implementation."""
    print("\nGenerating Side-by-Side Visual Parity Exhibits...")

    comparisons = [
        (
            MOCKUPS_DIR / "sparse_indic_editorial_academic_1793600154563.jpg",
            SCREENSHOTS_DIR / "screenshot_style2_academic.png",
            SCREENSHOTS_DIR / "parity_comparison_style2_academic.png",
            "Style 2: Clean Editorial Academic (Mockup vs Implemented HTML)"
        ),
        (
            MOCKUPS_DIR / "sparse_indic_bento_grid_1793600232944.jpg",
            SCREENSHOTS_DIR / "screenshot_style4_bento.png",
            SCREENSHOTS_DIR / "parity_comparison_style4_bento.png",
            "Style 4: Bento-Grid Dashboard (Mockup vs Implemented HTML)"
        ),
    ]

    for mockup_p, screenshot_p, out_p, title in comparisons:
        if not mockup_p.exists() or not screenshot_p.exists():
            print(f"Skipping {title}: missing files.")
            continue

        img_mock = Image.open(mockup_p).convert("RGB")
        img_real = Image.open(screenshot_p).convert("RGB")

        # Resize both to common height
        target_h = 1000
        w_mock = int(img_mock.width * (target_h / img_mock.height))
        w_real = int(img_real.width * (target_h / img_real.height))

        res_mock = img_mock.resize((w_mock, target_h), Image.Resampling.LANCZOS)
        res_real = img_real.resize((w_real, target_h), Image.Resampling.LANCZOS)

        combined = Image.new("RGB", (w_mock + w_real + 20, target_h + 60), (15, 23, 42))
        combined.paste(res_mock, (0, 60))
        combined.paste(res_real, (w_mock + 20, 60))

        combined.save(out_p)
        print(f"Saved side-by-side exhibit: {out_p} ({out_p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    capture_all_styles()
    create_side_by_side_parity_exhibits()
