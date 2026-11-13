"""train.py for Windows/Linux: the same fine-tune with PyTorch + PEFT on an NVIDIA GPU instead of MLX.

  python scripts/train_torch.py                        # configs/sft.yaml -> outputs/adapters_torch
  python scripts/train_torch.py --iters 10             # quick check
  python scripts/train_torch.py --summarize paper.pdf  # try the trained adapter on a paper
  python scripts/train_torch.py --merge                # adapter + base -> outputs/merged (to make a GGUF)

Reads the same data/processed/*.jsonl and settings as the Mac scripts, and keeps train.py's two
Gemma 4 fixes: thinking off in the prompt, and the loss (with the 262k-wide output head) only runs
on summary tokens. Runs one config; the `sweep:` list is Mac-only for now.
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig

from common import CONFIG, ROOT, messages, paper_text

OUT = ROOT / "outputs/adapters_torch"
CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "mps" if torch.backends.mps.is_available() else "cpu"


def load_base(four_bit: bool = True):
    """4-bit (QLoRA) needs bitsandbytes, which is NVIDIA-only; elsewhere it loads in 16-bit, for checks."""
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16) if four_bit and CUDA else None
    return AutoModelForMultimodalLM.from_pretrained(CONFIG["torch_model"], dtype=torch.bfloat16,
                                                    quantization_config=quant, device_map={"": DEVICE})


def add_lora(model):
    p, n = CONFIG["lora_parameters"], CONFIG["num_layers"]
    total = model.config.get_text_config().num_hidden_layers
    layers = "|".join(str(i) for i in range(total - n if n > 0 else 0, total))  # the last n layers
    keys = "|".join(re.escape(k) for k in p["keys"])
    # a regex, not a list of names: the vision and audio towers also have q_proj/v_proj and stay untouched
    target = rf".*language_model\.layers\.({layers})\.({keys})"
    return get_peft_model(model, LoraConfig(r=p["rank"], lora_alpha=p["scale"] * p["rank"],  # mlx scale = alpha / r
                                            lora_dropout=p["dropout"], target_modules=target))


def encode(tok, msgs: list[dict]) -> tuple[list[int], int]:
    """-> (token ids, index of the first summary token), thinking off like LM Studio chats."""
    full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert full.startswith(prompt), "chat template: the prompt is not a prefix of the full conversation"
    ids = tok.encode(full, add_special_tokens=False)[:CONFIG["max_seq_length"]]
    return ids, len(tok.encode(prompt, add_special_tokens=False))


def summary_loss(model, ids: list[int], offset: int) -> torch.Tensor:
    """Cross-entropy on the summary only. logits_to_keep runs the output head on just the last
    positions (the summary) instead of all ~10k."""
    x = torch.tensor([ids], device=DEVICE)
    keep = len(ids) - offset + 1  # positions offset-1 .. end; the last one has nothing left to predict
    with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=CUDA):
        logits = model(input_ids=x, logits_to_keep=keep, use_cache=False).logits[0, :-1]
    return torch.nn.functional.cross_entropy(logits.float(), x[0, offset:])


def load_rows(tok, split: str) -> list[tuple[list[int], int]]:
    lines = (ROOT / CONFIG["data"] / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    return [encode(tok, json.loads(line)["messages"]) for line in lines]


def train(iters: int):
    tok = AutoTokenizer.from_pretrained(CONFIG["torch_model"])
    rows, valid = load_rows(tok, "train"), load_rows(tok, "valid")
    model = add_lora(load_base())
    if CONFIG["grad_checkpoint"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # checkpointing drops the gradient unless its input requires grad
    model.print_trainable_parameters()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=CONFIG["learning_rate"])
    rng, order, seen, accum = random.Random(CONFIG["seed"]), [], [], CONFIG["grad_accumulation_steps"]
    # ponytail: one paper per step, batch_size in the yaml is ignored; ~10k-token papers don't batch anyway
    for it in range(1, iters + 1):
        order = order or rng.sample(rows, len(rows))  # new shuffled pass over the papers
        model.train()
        loss = summary_loss(model, *order.pop())
        (loss / accum).backward()
        seen.append(loss.item())
        if it % accum == 0 or it == iters:
            opt.step()
            opt.zero_grad()
        if it % CONFIG["steps_per_report"] == 0 or it == iters:
            print(f"iter {it}: train loss {sum(seen) / len(seen):.3f}", flush=True)
            seen = []
        if it % CONFIG["steps_per_eval"] == 0 or it == iters:
            model.eval()
            with torch.no_grad():
                print(f"iter {it}: valid loss {sum(summary_loss(model, *r).item() for r in valid) / len(valid):.3f}",
                      flush=True)
        if it % CONFIG["save_every"] == 0 or it == iters:
            model.save_pretrained(OUT)
    print(f"adapter saved to {OUT}")


def summarize(paper: Path):
    tok = AutoTokenizer.from_pretrained(CONFIG["torch_model"])
    model = PeftModel.from_pretrained(load_base(), OUT).eval()
    prompt = tok.apply_chat_template(messages(paper_text(paper)), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    x = tok(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**x, max_new_tokens=1500, do_sample=False)
    print(tok.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=True))


def merge():
    """Adapter + full-precision base -> a plain model folder, the input llama.cpp needs for a GGUF."""
    dest = ROOT / "outputs/merged"
    PeftModel.from_pretrained(load_base(four_bit=False), OUT).merge_and_unload().save_pretrained(dest)
    AutoTokenizer.from_pretrained(CONFIG["torch_model"]).save_pretrained(dest)
    print(f"merged model saved to {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=CONFIG["iters"])
    ap.add_argument("--summarize", type=Path, metavar="PAPER")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge()
    elif args.summarize:
        summarize(args.summarize)
    else:
        train(args.iters)
