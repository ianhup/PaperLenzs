"""Train every entry under `sweep:` in configs/sft.yaml, score each on the valid split,
copy the winner into adapter_path (where eval.py and export.py look).

  python scripts/sweep.py                 # all runs
  python scripts/sweep.py --iters 20      # extra flags go to every training run (quick check)
"""
import shutil
import subprocess
import sys

import yaml

from common import CONFIG, ROOT
from eval import evaluate, rank_key, report


def merged(overrides: dict) -> dict:
    cfg = {k: v for k, v in CONFIG.items() if k != "sweep"}
    for k, v in overrides.items():
        cfg[k] = {**cfg[k], **v} if isinstance(v, dict) else v
    return cfg


results = []
for name, overrides in CONFIG["sweep"].items():
    run = ROOT / "outputs/runs" / name
    run.mkdir(parents=True, exist_ok=True)
    cfg = merged(overrides) | {"adapter_path": str(run)}
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    print(f"\n===== training {name}: {overrides}", flush=True)
    # separate process per run so each one starts with all RAM free
    subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "-c", str(run / "config.yaml"),
                    *sys.argv[1:]], check=True)
    results.append(evaluate(run, "valid", name))

report(results)
best = max(results, key=rank_key)
dest = ROOT / CONFIG["adapter_path"]
shutil.copytree(ROOT / "outputs/runs" / best["label"], dest, dirs_exist_ok=True)
print(f"\nbest: {best['label']} -> copied to {dest}. Next: python scripts/eval.py (test set, base vs tuned)")
