"""mlx_lm.lora with configs/sft.yaml, plus two fixes for Gemma 4 on long papers.

1. Thinking OFF when formatting training data. mlx-lm turns Gemma 4's thinking mode on by
   default, LM Studio chats with it off; training in one layout and chatting in the other
   weakens the fine-tune.
2. Summary-only loss. Gemma's vocabulary is 262k tokens; the stock loss scores every vocab
   entry at every position (paper included), ~9 GB at 9k tokens -> out of memory on 24 GB.
   With mask_prompt only summary positions count, so only those get scored.

Extra CLI flags pass through, e.g. `python scripts/train.py --iters 10`.
"""
import sys
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import lora
from mlx_lm.tuner import trainer
from mlx_lm.tokenizer_utils import TokenizerWrapper

from common import CONFIG, ROOT

_apply = TokenizerWrapper.apply_chat_template


def _no_thinking(self, *args, **kwargs):
    kwargs.setdefault("enable_thinking", False)
    return _apply(self, *args, **kwargs)


TokenizerWrapper.apply_chat_template = _no_thinking


def summary_only_loss(model, batch, lengths, width=CONFIG["max_summary_tokens"]):
    """Same value as mlx's default_loss with mask_prompt, but the output head only runs on a
    fixed-width window starting at the summary (fixed width because the step is mx.compiled)."""
    inputs, targets = batch[:, :-1], batch[:, 1:]
    lm = getattr(model, "language_model", model)
    hidden = lm.model(inputs)
    start = lengths[:, :1] - 1                      # first position whose target is summary
    pos = start + mx.arange(width)[None]
    mask = pos < lengths[:, 1:]                     # inside the real (unpadded) sequence
    pos = mx.minimum(pos, targets.shape[1] - 1)
    hidden = mx.take_along_axis(hidden, pos[..., None], axis=1)
    targets = mx.take_along_axis(targets, pos, axis=1)
    logits = lm.model.embed_tokens.as_linear(hidden) if lm.tie_word_embeddings else lm.lm_head(hidden)
    if cap := lm.final_logit_softcapping:
        logits = mx.tanh(logits / cap) * cap
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks


if __name__ == "__main__":
    lora.train = partial(trainer.train, loss=summary_only_loss)
    sys.argv[1:1] = ["-c", str(ROOT / "configs/sft.yaml")]
    lora.main()
