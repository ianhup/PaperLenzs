"""Run: python tests/test_loss.py   (loads the base model, ~20 s)

The two things in train.py that fail silently instead of crashing:
  - the summary-only loss must equal mlx's stock masked loss exactly
  - training data must be formatted with thinking off and the mask must start at the summary
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.tuner.datasets import ChatDataset  # noqa: E402
from mlx_lm.tuner.trainer import default_loss  # noqa: E402

from common import CONFIG, SECTIONS, messages  # noqa: E402
from train import summary_only_loss  # noqa: E402

model, tok = load(CONFIG["model"])
summary = "\n\n".join(f"## {s}\nSoil bacteria fell 12.4 percent under drought." for s in SECTIONS)
row = {"messages": messages("We dried 40 grassland plots for 90 days. " * 30, summary)}
ids, offset = ChatDataset([row], tok, mask_prompt=True).process(row)

text = tok.decode(ids)
assert "<|think|>" not in text, "training data has thinking on"
assert tok.decode(ids[offset:]).startswith("## Title"), "loss mask does not start at the summary"

# pad like mlx's batcher does, so the padded tail must be ignored too
batch = mx.array(ids + [0] * 7)[None]
lengths = mx.array([[offset, len(ids)]])
stock, n1 = default_loss(model, batch, lengths)
ours, n2 = summary_only_loss(model, batch, lengths)
assert n1.item() == n2.item(), (n1.item(), n2.item())
assert abs(stock.item() - ours.item()) < 1e-3, (stock.item(), ours.item())

# summary longer than the window -> tokens past max_summary_tokens are dropped, so loss differs
_, n3 = summary_only_loss(model, batch, lengths, width=8)
assert n3.item() == 8

print(f"ok  (loss {ours.item():.4f} == stock {stock.item():.4f}, {n2.item()} summary tokens)")
