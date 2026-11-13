# Paper summarizer fine-tune (Gemma 4 E2B, MLX, LM Studio)

Teaches Gemma 4 E2B to answer every paper with the same 8 sections (Title, Topics,
Introduction, Methods, Results, Discussion, Conclusion, Relevance to research), then puts the
model into LM Studio, where you can keep asking questions about the paper.

## Setup (once, Mac; for Windows/Linux see [below](#windows--linux-nvidia-gpu))

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r pyproject.toml
```

## Each time you have new data

1. **Import each teammate's CSV** (columns `Input` = pdf path, `Output` = summary) with the
   folder of PDFs it names. One set per research project:

```bash
.venv/bin/python scripts/import_csv.py summaries.csv "folder with the pdfs" set_name
```

   That writes `data/raw/set_name/<paper>.pdf` + `<paper>.summary.md` pairs
   (see [examples/](examples/) for the shape; you can also add pairs by hand).
2. **Write that set's research into `data/raw/set_name/research.md`**. The set's Relevance
   sections are trained against this text, so it has to be the research they were written for.
   Pairs in a folder without a `research.md` use `My research:` from
   [prompts/system.md](prompts/system.md), which is also what you fill in for LM Studio.
3. Run:

```bash
.venv/bin/python scripts/prepare_data.py   # checks every summary, splits train/valid/test
.venv/bin/python scripts/sweep.py          # trains each `sweep:` config, keeps the best on valid
.venv/bin/python scripts/eval.py           # base vs fine-tuned on the untouched test set
.venv/bin/python scripts/export.py         # merge + copy into LM Studio
```

4. In LM Studio, load `local/paper-summarizer`, paste `prompts/system.md` as the system
   prompt (save it as a preset), keep thinking off, and drop in a paper.

Other commands:
- `scripts/train.py`: one training run with `configs/sft.yaml` only, no sweep
- `scripts/summarize.py paper.pdf [--base]`: summarize from the terminal
- `tests/test_format.py`, `tests/test_loss.py`: run after changing the format or upgrading mlx-lm

## Windows / Linux (NVIDIA GPU)

MLX only runs on Apple Silicon, so here training uses PyTorch instead: same data, same
`configs/sft.yaml`, same prompt. Setup once (PowerShell; on Linux the Python is `.venv/bin/python`):

```powershell
git config --global core.longpaths true   # some paper file names are long
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r pyproject.toml --torch-backend=auto
```

Then:

```powershell
.venv\Scripts\python.exe scripts\prepare_data.py                        # same as on Mac
.venv\Scripts\python.exe scripts\train_torch.py                         # one run -> outputs\adapters_torch
.venv\Scripts\python.exe scripts\train_torch.py --summarize paper.pdf   # try it on a paper
```

- `--iters 10` is a quick check; `tests\test_torch.py` checks the loss and the LoRA placement.
- The first run downloads the base model (`torch_model` in the config, 10 GB).
- Out of GPU memory: lower `max_seq_length` in `configs/sft.yaml` and re-run `prepare_data.py`.
  The Mac numbers under Limits don't carry over; the base model's embedding tables alone stay
  16-bit and take about 5.5 GB.
- `sweep.py`, `eval.py` and `export.py` are Mac-only. For LM Studio on Windows,
  `train_torch.py --merge` writes the fine-tuned model to `outputs\merged`; convert that folder to
  GGUF with llama.cpp's `convert_hf_to_gguf.py` and load the `.gguf` in LM Studio.

## Files

| file | what it does |
|---|---|
| `configs/sft.yaml` | every setting: base model, LoRA, lengths, the sweep list |
| `prompts/system.md` | the instructions + your research profile; same text in training and LM Studio |
| `scripts/common.py` | section list, format checker, made-up-number checker, PDF reading |
| `scripts/import_csv.py` | a teammate's CSV + PDFs → `data/raw/<set>/` pairs, headings normalized |
| `scripts/prepare_data.py` | raw pairs → `data/processed/{train,valid,test}.jsonl`, trims long papers to fit |
| `scripts/train.py` | `mlx_lm.lora` + two Gemma 4 fixes (thinking off, summary-only loss) |
| `scripts/train_torch.py` | the same training for Windows/Linux (PyTorch + PEFT), plus `--summarize` and `--merge` |
| `scripts/sweep.py` | runs every sweep config, picks the winner by format → made-up numbers → loss |
| `scripts/eval.py` | generates summaries, writes them to `outputs/eval/` for you to read |
| `scripts/export.py` | fuses adapter → `~/.lmstudio/models/local/<export_name>` |

## Limits (measured on a 24 GB M4 Pro)

- **Training reads at most ~10k tokens per paper** (`max_seq_length: 10240`). Longer papers are
  trimmed for training only, with a warning. In LM Studio the model still reads the whole paper.
  Memory: 8k tokens = 12 GB, 12k = 19 GB, over the GPU limit.
- **Speed:** ~35 s per step at 10k tokens. 300 steps is about 3 h per run, so the default
  3-run sweep is an overnight job. Fewer sweep entries or `--iters` cut that.
- **Settings found by testing:** LoRA `scale: 4` (mlx's default of 20 broke Gemma 4 within 10 steps),
  attention + MLP layers only, last 16 layers.

## About DPO (steps 3–4 of the SFT → DPO guide)

Not included yet. This template covers the guide's steps 1, 2 and 5: several SFT configs, pick
the best on a dev set, final base-vs-tuned comparison on a held-out test set. DPO needs pairs of
a *chosen* and a *rejected* answer for the same paper. You'll get those naturally once SFT runs:
read the summaries in `outputs/eval/`, write corrected versions of the bad ones, and each
(corrected, bad) pair is a DPO example. mlx-lm itself has no DPO trainer, so that step would
add a separate package.
