"""Deploy Style 2 (Editorial Academic) to GitHub Pages.

Builds a self-contained static site with Style 2 as index.html,
preserves Style 1 (glassmorphism.html) and Style 4 (bento.html),
and pushes to the remote gh-pages branch.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo"
BUILD_DIR = REPO_ROOT / ".gh_pages_build"


def build_site() -> None:
    print("=" * 70)
    print("Building GitHub Pages Site (Style 2: Editorial Academic Default)...")
    print("=" * 70)

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy assets, data, figures, screenshots
    for folder in ["assets", "data", "figures", "screenshots"]:
        src_folder = DEMO_DIR / folder
        dst_folder = BUILD_DIR / folder
        if src_folder.exists():
            shutil.copytree(src_folder, dst_folder)
            print(f"Copied {folder}/ to build directory.")

    # 2. Add .nojekyll
    (BUILD_DIR / ".nojekyll").touch()
    print("Created .nojekyll file.")

    # 3. Prepare Style 2 as index.html and academic.html
    academic_src = DEMO_DIR / "academic.html"
    academic_content = academic_src.read_text(encoding="utf-8")

    # Update style switcher in academic.html so index.html is the active link
    acad_switcher_target = '<a href="academic.html" class="style-btn active">Editorial Academic (Style 2)</a>'
    acad_switcher_replacement = '<a href="index.html" class="style-btn active">Editorial Academic (Style 2)</a>'
    academic_content = academic_content.replace(acad_switcher_target, acad_switcher_replacement)
    academic_content = academic_content.replace('<a href="index.html" class="style-btn">Dark Glassmorphism (Style 1)</a>', '<a href="glassmorphism.html" class="style-btn">Dark Glassmorphism (Style 1)</a>')

    (BUILD_DIR / "index.html").write_text(academic_content, encoding="utf-8")
    (BUILD_DIR / "academic.html").write_text(academic_content, encoding="utf-8")
    print("Prepared index.html and academic.html (Style 2: Editorial Academic).")

    # 4. Prepare Style 1 as glassmorphism.html
    glass_src = DEMO_DIR / "index.html"
    glass_content = glass_src.read_text(encoding="utf-8")
    glass_content = glass_content.replace('href="index.html" style="background: rgba(16, 185, 129, 0.15)', 'href="glassmorphism.html" style="background: rgba(16, 185, 129, 0.15)')
    glass_content = glass_content.replace('href="academic.html"', 'href="index.html"')
    (BUILD_DIR / "glassmorphism.html").write_text(glass_content, encoding="utf-8")
    print("Prepared glassmorphism.html (Style 1: Dark Glassmorphism).")

    # 5. Prepare Style 4 as bento.html
    bento_src = DEMO_DIR / "bento.html"
    bento_content = bento_src.read_text(encoding="utf-8")
    bento_content = bento_content.replace('<a href="index.html" class="style-pill">Dark Glassmorphism (Style 1)</a>', '<a href="glassmorphism.html" class="style-pill">Dark Glassmorphism (Style 1)</a>')
    bento_content = bento_content.replace('<a href="academic.html" class="style-pill">Editorial Academic (Style 2)</a>', '<a href="index.html" class="style-pill">Editorial Academic (Style 2)</a>')
    (BUILD_DIR / "bento.html").write_text(bento_content, encoding="utf-8")
    print("Prepared bento.html (Style 4: Bento-Grid Dashboard).")

    # 6. Add Readme
    readme_text = """# SPARSE-INDIC Interactive Tokenizer Playground
Hosted on GitHub Pages at: https://rohzhegde26.github.io/NLP_TOKENIZER_TAX/

- **Default Layout:** Style 2: Clean Editorial Academic (`index.html` / `academic.html`)
- **Alternative Style:** Style 1: Dark Glassmorphism (`glassmorphism.html`)
- **Alternative Style:** Style 4: Modern Bento-Grid Dashboard (`bento.html`)
"""
    (BUILD_DIR / "README.md").write_text(readme_text, encoding="utf-8")
    print("Prepared site README.md.")


def push_to_gh_pages() -> None:
    print("\nPushing site build to 'gh-pages' branch...")
    env = os.environ.copy()
    temp_index = REPO_ROOT / ".git_gh_pages_index"
    env["GIT_INDEX_FILE"] = str(temp_index)
    env["GIT_WORK_TREE"] = str(BUILD_DIR)

    if temp_index.exists():
        temp_index.unlink()

    try:
        # Stage all files in BUILD_DIR
        subprocess.run(["git", "add", "-A"], cwd=str(BUILD_DIR), env=env, check=True)

        # Write tree
        tree_res = subprocess.run(["git", "write-tree"], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=True)
        tree_sha = tree_res.stdout.strip()
        print(f"Created Git Tree: {tree_sha}")

        # Commit tree
        commit_msg = "deploy: publish SPARSE-INDIC Style 2 Editorial Academic to GitHub Pages"
        commit_res = subprocess.run(["git", "commit-tree", tree_sha, "-m", commit_msg], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=True)
        commit_sha = commit_res.stdout.strip()
        print(f"Created Git Commit: {commit_sha}")

        # Update local gh-pages branch
        subprocess.run(["git", "update-ref", "refs/heads/gh-pages", commit_sha], cwd=str(REPO_ROOT), check=True)
        print("Updated local branch 'gh-pages'.")

        # Push to remote origin gh-pages
        print("Pushing 'gh-pages' to origin...")
        subprocess.run(["git", "push", "-u", "origin", "gh-pages", "--force"], cwd=str(REPO_ROOT), check=True)
        print("[SUCCESS] Successfully pushed 'gh-pages' branch to GitHub!")

    finally:
        if temp_index.exists():
            temp_index.unlink()
        if BUILD_DIR.exists():
            shutil.rmtree(BUILD_DIR)


if __name__ == "__main__":
    build_site()
    push_to_gh_pages()
