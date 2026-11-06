"""Merge the adapter into the base model and drop it into LM Studio's model folder (MLX format,
LM Studio runs it natively on Apple Silicon). Then load it in LM Studio and paste prompts/system.md
as the system prompt."""
import shutil
import subprocess
import sys
from pathlib import Path

from common import CONFIG, ROOT

dest = Path.home() / ".lmstudio/models/local" / CONFIG["export_name"]
if dest.exists():
    shutil.rmtree(dest)  # previous export of this same name; re-exporting replaces it
subprocess.run([sys.executable, "-m", "mlx_lm", "fuse", "--model", CONFIG["model"],
                "--adapter-path", str(ROOT / CONFIG["adapter_path"]), "--save-path", str(dest)],
               check=True)
print(f"\nExported to {dest}\nIn LM Studio: load 'local/{CONFIG['export_name']}', "
      f"set the system prompt to the contents of prompts/system.md, thinking off.")
