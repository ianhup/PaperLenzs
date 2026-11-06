"""data/raw/<name>.pdf (or .txt/.md) + data/raw/<name>.summary.md -> data/processed/{train,valid,test}.jsonl

Bad summaries (wrong sections) stop the run; made-up-looking numbers only warn.
Papers too long for max_seq_length are cut so the summary (the thing we train on) is never cut.
"""
import json
import random

from transformers import AutoTokenizer

from common import (CONFIG, ROOT, SYSTEM_PROMPT, format_problems, messages, paper_text,
                    unsupported_numbers)

RAW, OUT = ROOT / "data/raw", ROOT / CONFIG["data"]
HOLDOUT_FRACTION = 0.1  # each of valid and test


def main():
    if "EDIT ME" in SYSTEM_PROMPT:
        raise SystemExit("Fill in 'My research:' in prompts/system.md first; every summary's "
                         "'Relevance to research' section is trained against it.")
    tok = AutoTokenizer.from_pretrained(CONFIG["model"])
    budget = CONFIG["max_seq_length"]
    rows, errors = [], []

    for summary_path in sorted(RAW.glob("*.summary.md")):
        name = summary_path.name.removesuffix(".summary.md")
        papers = [p for ext in (".pdf", ".txt", ".md") if (p := RAW / f"{name}{ext}").exists()]
        if not papers:
            errors.append(f"{name}: no {name}.pdf/.txt/.md next to the summary")
            continue
        summary, paper = summary_path.read_text().strip(), paper_text(papers[0])

        if problems := format_problems(summary):
            errors.append(f"{name}: {'; '.join(problems)}")
            continue
        if nums := unsupported_numbers(summary, paper):
            print(f"warn {name}: numbers not found in paper: {nums}")

        summary_tokens = len(tok.encode(summary, add_special_tokens=False)) + 2  # + end-of-turn
        if summary_tokens > CONFIG["max_summary_tokens"]:
            errors.append(f"{name}: summary is {summary_tokens} tokens, over max_summary_tokens "
                          f"{CONFIG['max_summary_tokens']} (shorten it or raise the limit)")
            continue

        # fit paper into the token budget: everything except the paper is fixed cost
        layout = tok.apply_chat_template(messages("", summary), tokenize=False, enable_thinking=False)
        fixed = len(tok.encode(layout, add_special_tokens=False))
        paper_ids = tok.encode(paper, add_special_tokens=False)
        room = budget - fixed - 16
        if room < 1000:
            errors.append(f"{name}: summary alone is ~{fixed} tokens, too long for max_seq_length {budget}")
            continue
        if len(paper_ids) > room:
            print(f"warn {name}: paper cut {len(paper_ids)} -> {room} tokens (raise max_seq_length to keep more)")
            paper = tok.decode(paper_ids[:room])
        rows.append({"messages": messages(paper, summary)})

    if errors:
        raise SystemExit("Fix these first:\n  " + "\n  ".join(errors))
    if len(rows) < 3:
        raise SystemExit(f"Need at least 3 paper+summary pairs in {RAW}, found {len(rows)}")

    # valid = picks the best sweep run; test = untouched until the final base-vs-tuned comparison
    random.Random(CONFIG["seed"]).shuffle(rows)
    n = max(1, round(len(rows) * HOLDOUT_FRACTION))
    splits = {"test": rows[:n], "valid": rows[n:2 * n], "train": rows[2 * n:]}
    OUT.mkdir(parents=True, exist_ok=True)
    for split, part in splits.items():
        (OUT / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
    print("ok: " + ", ".join(f"{len(p)} {s}" for s, p in splits.items()) + f" -> {OUT}")


if __name__ == "__main__":
    main()
