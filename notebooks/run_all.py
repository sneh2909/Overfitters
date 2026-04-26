#!/usr/bin/env python
"""
run_all.py — run the four notebooks as a single pipeline, headless.

Equivalent to run_all.sh but pure-Python and easier to integrate into CI.

Usage:
    HF_TOKEN=hf_... HF_USERNAME=SnehShah python notebooks/run_all.py [--stage all|explore|sft|grpo|eval]

Requirements:
    pip install nbclient nbformat
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
PIPELINE = {
    "explore": "01_explore_env.ipynb",
    "sft":     "02_sft.ipynb",
    "grpo":    "03_grpo.ipynb",
    "eval":    "04_eval_compare.ipynb",
}


def execute(nb_path: Path, timeout: int = 3600) -> None:
    """Execute a notebook in-place; results land at executed_<name>.ipynb."""
    import nbformat
    from nbclient import NotebookClient

    print()
    print("=" * 60)
    print(f" running  {nb_path.name}")
    print("=" * 60)
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
    try:
        client.execute()
    except Exception as exc:
        print(f"\n[ERROR] {nb_path.name} raised: {exc!r}", file=sys.stderr)
        raise
    out_path = nb_path.with_name(f"executed_{nb_path.name}")
    nbformat.write(nb, out_path)
    print(f"  saved \u2192 {out_path.relative_to(Path.cwd())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=list(PIPELINE.keys()) + ["all"],
                    help="which notebook(s) to run (default: all)")
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("warning: HF_TOKEN not set \u2014 push steps will be skipped", file=sys.stderr)

    targets = list(PIPELINE.values()) if args.stage == "all" else [PIPELINE[args.stage]]
    for name in targets:
        execute(NB_DIR / name)

    print("\nall done. executed notebooks saved as notebooks/executed_*.ipynb")


if __name__ == "__main__":
    main()
