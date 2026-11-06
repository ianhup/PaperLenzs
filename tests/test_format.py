"""Run: python tests/test_format.py  (no test framework needed)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import SECTIONS, SYSTEM_PROMPT, format_problems, unsupported_numbers  # noqa: E402

good = "\n\n".join(f"## {s}\nsome text" for s in SECTIONS)
assert format_problems(good) == []
assert format_problems(good.replace("## Methods", "## Method"))            # renamed section
assert format_problems(good.replace("## Results\nsome text", "## Results\n "))  # empty section
swapped = good.replace("## Results", "## TMP").replace("## Methods", "## Results").replace("## TMP", "## Methods")
assert format_problems(swapped)                                             # wrong order
assert format_problems("Sure! Here it is:\n\n" + good)                     # chatter before the summary

assert unsupported_numbers("accuracy 93.4 on 120 images", "we got 93.4 on 120 images") == []
assert unsupported_numbers("accuracy 97.1", "we got 93.4") == ["97.1"]

for s in SECTIONS:  # the prompt and the checker must agree on the format
    assert f"## {s}" in SYSTEM_PROMPT, s

print("ok")
