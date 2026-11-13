"""Run: python tests/test_torch.py   (loads the PyTorch base model; Windows/Linux counterpart of test_loss.py)

The things in train_torch.py that fail silently instead of crashing:
  - the prompt must be formatted with thinking off and the loss must start at the summary
  - the summary-only loss must equal the model's own loss with the prompt masked out
  - LoRA must land on the text layers only, and gradients must reach it
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from common import CONFIG, SECTIONS, messages  # noqa: E402
from train_torch import DEVICE, add_lora, encode, load_base, summary_loss  # noqa: E402

tok = AutoTokenizer.from_pretrained(CONFIG["torch_model"])
summary = "\n\n".join(f"## {s}\nSoil bacteria fell 12.4 percent under drought." for s in SECTIONS)
ids, offset = encode(tok, messages("We dried 40 grassland plots for 90 days. " * 30, summary))
assert "<|think|>" not in tok.decode(ids), "training data has thinking on"
assert tok.decode(ids[offset:]).startswith("## Title"), "loss does not start at the summary"

model = load_base().eval()
x = torch.tensor([ids], device=DEVICE)
labels = x.clone()
labels[0, :offset] = -100  # the stock way: score every position, ignore the prompt
with torch.no_grad():
    stock, ours = model(input_ids=x, labels=labels).loss.item(), summary_loss(model, ids, offset).item()
assert abs(stock - ours) < 2e-2, (stock, ours)

model = add_lora(model)
adapted = [n for n, _ in model.named_modules() if n.endswith("lora_A")]
assert adapted and all("language_model" in n for n in adapted), "LoRA is not only on the text layers"
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.train()
summary_loss(model, ids, offset).backward()
grads = [p.grad for n, p in model.named_parameters() if "lora_B" in n]
assert grads and all(g is not None and g.abs().sum() > 0 for g in grads), "no gradient reaches LoRA"

print(f"ok  (loss {ours:.4f} == stock {stock:.4f}, {len(ids) - offset} summary tokens, {len(adapted)} LoRA modules)")
