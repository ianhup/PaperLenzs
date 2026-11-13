"""Team CSV (columns: Input = pdf path, Output = summary) + the folder of PDFs -> data/raw/<set>/ pairs.

  python scripts/import_csv.py <summaries.csv> <pdf folder> <set name>
Then write that set's research into data/raw/<set>/research.md and run prepare_data.py.
"""
import csv
import re
import shutil
import sys
import unicodedata
from pathlib import Path

from common import ROOT, SECTIONS, format_problems

_LABEL = re.compile(r"^(?:##[ \t]*)?(%s)(?::|[ \t]*$)[ \t]*" % "|".join(SECTIONS), re.M)


def to_markdown(output: str) -> str:
    """'Title: x', 'Title:\\nx' and '## Title\\nx' all become '## Title\\nx'."""
    parts = _LABEL.split(output.strip())
    return parts[0] + "\n\n".join(f"## {h}\n{b.strip()}" for h, b in zip(parts[1::2], parts[2::2]))


def main(csv_path: str, pdf_dir: str, name: str):
    out = ROOT / "data/raw" / name
    out.mkdir(parents=True, exist_ok=True)
    nfc = lambda s: unicodedata.normalize("NFC", s)  # noqa: E731  macOS file names vs CSV text
    pdfs = {nfc(p.name): p for p in Path(pdf_dir).rglob("*.pdf")}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))[1:]
    done, bad = 0, []
    for inp, output, *_ in rows:
        file = nfc(re.split(r"[\\/]", inp)[-1])  # CSVs carry /mnt/data/... and C:\... paths
        summary = to_markdown(output)
        if file not in pdfs:
            bad.append(f"{file}: no such pdf in {pdf_dir}")
        elif problems := format_problems(summary):
            bad.append(f"{file}: {problems[0]}")
        else:
            src = pdfs[file]
            shutil.copy2(src, out / src.name)
            (out / f"{src.stem}.summary.md").write_text(summary + "\n")
            done += 1
    print(f"{name}: {done}/{len(rows)} pairs -> {out}")
    if not (out / "research.md").exists():
        print(f"next: write this set's research into {out / 'research.md'}")
    if bad:
        raise SystemExit("skipped:\n  " + "\n  ".join(bad))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
