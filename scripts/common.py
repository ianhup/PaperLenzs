"""Shared by every script: config, paper loading, prompt layout, output format."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
# encoding="utf-8" everywhere: Windows would otherwise read these files as cp1252
CONFIG = yaml.safe_load((ROOT / "configs/sft.yaml").read_text(encoding="utf-8"))
_INSTRUCTIONS, _, _RESEARCH = ((ROOT / "prompts/system.md").read_text(encoding="utf-8").strip()
                               .partition("My research:"))


def system_prompt(research: str = _RESEARCH) -> str:
    """Instructions + the research that 'Relevance to research' is judged against."""
    return f"{_INSTRUCTIONS}My research:\n{research.strip()}"


SYSTEM_PROMPT = system_prompt()  # prompts/system.md as written: what you paste into LM Studio
USER_PREFIX = "Summarize this paper.\n\n"


def paper_text(path: Path) -> str:
    """PDF/txt/md -> plain text, reference list dropped (wastes tokens, no signal)."""
    if path.suffix.lower() == ".pdf":
        import pymupdf  # pypdf crashed on 2 of the first 149 papers; this reads all of them
        with pymupdf.open(path) as doc:
            text = "\n".join(page.get_text() for page in doc)
    else:
        text = path.read_text(encoding="utf-8")
    refs = [m.start() for m in re.finditer(r"^\s*(references|bibliography)\s*$", text, re.I | re.M)]
    if refs and refs[-1] > len(text) / 2:
        text = text[:refs[-1]]
    return text.strip()


def messages(paper: str, summary: str | None = None, system: str = SYSTEM_PROMPT) -> list[dict]:
    """The exact chat layout used in training; inference must match it."""
    m = [{"role": "system", "content": system},
         {"role": "user", "content": USER_PREFIX + paper}]
    return m + [{"role": "assistant", "content": summary}] if summary else m

SECTIONS = ["Title", "Topics", "Introduction", "Methods", "Results",
            "Discussion", "Conclusion", "Relevance to research"]

_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.M)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def format_problems(summary: str) -> list[str]:
    """Empty list = summary has exactly SECTIONS, in order, each non-empty."""
    found = _HEADING.findall(summary)
    if not summary.lstrip().startswith(f"## {SECTIONS[0]}"):
        return ["text before the first heading"]
    if found != SECTIONS:
        return [f"headings {found} != {SECTIONS}"]
    bodies = _HEADING.split(summary)[2::2]
    return [f"empty section: {s}" for s, b in zip(SECTIONS, bodies) if not b.strip()]


def unsupported_numbers(summary: str, paper: str) -> list[str]:
    """Numbers in the summary that never appear in the paper (likely made up)."""
    # ponytail: plain string match, misses reformatted numbers (0.5 vs 50%); fine as a warning
    paper_nums = set(_NUMBER.findall(paper))
    return sorted({n for n in _NUMBER.findall(summary) if n not in paper_nums and len(n) > 1})
