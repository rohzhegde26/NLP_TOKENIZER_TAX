"""Every number macro used in the paper must be defined in paper/numbers.tex."""
import os
import re

import pytest

pytestmark = pytest.mark.skipif(not os.path.exists("paper/numbers.tex"), reason="numbers.tex not generated")
PREFIX = r"(?:tax|fert|fertratio|sb|inr|meanTax|maxTax|maxTaxLang|kWin|taxCut|nTokenizers|nLangs)"


def test_all_paper_macros_defined():
    defs = set(re.findall(r"\\newcommand\{\\(\w+)\}", open("paper/numbers.tex", encoding="utf-8").read()))
    used = set()
    for f in ["paper/main.tex", "paper/sections/methodology_e1_e2.tex"]:
        used |= set(re.findall(r"\\(" + PREFIX + r"\w*)", open(f, encoding="utf-8").read()))
    assert used, "no macros found"
    assert used <= defs, f"undefined: {sorted(used - defs)}"
