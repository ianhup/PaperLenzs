"""Summarize papers from the terminal with the same prompt layout used in training.

  python scripts/summarize.py paper.pdf [more.pdf ...]            # base model + your adapter
  python scripts/summarize.py paper.pdf --base                    # base model only, to compare
"""
import argparse
from pathlib import Path

from mlx_lm import generate, load

from common import CONFIG, ROOT, messages, paper_text


def load_model(adapter: str | Path | None = ROOT / CONFIG["adapter_path"]):
    """adapter=None -> plain base model."""
    return load(CONFIG["model"], adapter_path=str(adapter) if adapter else None)


def summarize(model, tok, paper: str, max_tokens: int = 1500) -> str:
    prompt = tok.apply_chat_template(messages(paper), add_generation_prompt=True,
                                     enable_thinking=False)
    return generate(model, tok, prompt, max_tokens=max_tokens)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("papers", nargs="+", type=Path)
    ap.add_argument("--base", action="store_true", help="skip the adapter")
    args = ap.parse_args()
    model, tok = load_model(None if args.base else ROOT / CONFIG["adapter_path"])
    for p in args.papers:
        print(f"\n===== {p.name} =====\n{summarize(model, tok, paper_text(p))}")
