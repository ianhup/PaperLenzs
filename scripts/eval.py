"""Score a model on a split by generating summaries and checking them like the training data.

  python scripts/eval.py                 # final comparison on test: base vs fine-tuned
  python scripts/eval.py --split valid   # same on valid
Summaries are written to outputs/eval/<split>/<model>/ so you can read them.
"""
import argparse
import json

import mlx.core as mx
from mlx_lm.tuner.datasets import ChatDataset

from train import summary_only_loss  # also applies the thinking-off patch
from common import CONFIG, ROOT, USER_PREFIX, format_problems, unsupported_numbers
from summarize import load_model, summarize


def summary_loss(model, tok, row) -> float:
    """Loss on the reference summary only, same masking as training. Lower = closer to your style."""
    ids, offset = ChatDataset([row], tok, mask_prompt=True).process(row)
    loss, _ = summary_only_loss(model, mx.array(ids)[None], mx.array([[offset, len(ids)]]))
    return loss.item()


def evaluate(adapter, split: str, label: str) -> dict:
    """adapter=None scores the base model. Ranking: format pass, then made-up numbers, then loss."""
    model, tok = load_model(adapter)
    out = ROOT / "outputs/eval" / split / label
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (ROOT / CONFIG["data"] / f"{split}.jsonl").read_text().splitlines()]
    passed = bad_numbers = 0
    loss = sum(summary_loss(model, tok, r) for r in rows) / len(rows)
    for i, row in enumerate(rows):
        paper = row["messages"][1]["content"].removeprefix(USER_PREFIX)
        got = summarize(model, tok, paper, row["messages"][0]["content"])  # that row's own research
        (out / f"{i}.md").write_text(got)
        problems, nums = format_problems(got), unsupported_numbers(got, paper)
        passed += not problems
        bad_numbers += len(nums)
        print(f"  [{label} {i}] format {'OK' if not problems else problems[0][:100]} | made-up numbers: {nums[:8] or 'none'}")
    return {"label": label, "format_pass": passed / len(rows), "made_up_numbers": bad_numbers,
            "loss": loss, "n": len(rows)}


def rank_key(r: dict):
    return (r["format_pass"], -r["made_up_numbers"], -r["loss"])


def report(results: list[dict]):
    print(f"\n{'model':<24}{'format pass':>12}{'made-up numbers':>18}{'summary loss':>14}")
    for r in results:
        print(f"{r['label']:<24}{r['format_pass']:>11.0%}{r['made_up_numbers']:>18}{r['loss']:>14.3f}   (n={r['n']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    split = ap.parse_args().split
    report([evaluate(None, split, "base"), evaluate(ROOT / CONFIG["adapter_path"], split, "fine-tuned")])
