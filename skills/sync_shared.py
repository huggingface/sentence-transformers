#!/usr/bin/env python3
"""Mirror shared reference docs + hard-negative mining script across the three
sentence-transformers skills.

The three skills under `skills/` are each fully self-contained per Hugging Face
skill convention, but a handful of files (cross-cutting reference docs + the
hard-negative mining CLI) are duplicated across all three. To avoid drift, edit
the canonical copy under `skills/train-sentence-transformer/` and run this
script — it copies the canonical version into `train-cross-encoder` and
`train-sparse-encoder`.

Usage:
    python skills/sync_shared.py            # sync (writes files)
    python skills/sync_shared.py --check    # read-only; exit 1 on drift

Wired up as a pre-commit hook (auto-fix on commit) and verified by `make check`
in CI via the same hook.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SKILLS_DIR.parent
CANONICAL = "train-sentence-transformer"
OTHERS = ("train-cross-encoder", "train-sparse-encoder")
SHARED_FILES = (
    "references/dataset_formats.md",
    "references/hardware_guide.md",
    "references/hf_jobs_execution.md",
    "references/prompts_and_instructions.md",
    "references/training_args.md",
    "references/troubleshooting.md",
    "scripts/mine_hard_negatives.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Don't modify files; exit 1 if any drift is detected.",
    )
    args = parser.parse_args()

    drift: list[Path] = []
    for rel in SHARED_FILES:
        src = SKILLS_DIR / CANONICAL / rel
        if not src.is_file():
            sys.stderr.write(f"ERROR: canonical file missing: {src.relative_to(REPO_ROOT)}\n")
            return 2
        for other in OTHERS:
            dst = SKILLS_DIR / other / rel
            if dst.is_file() and filecmp.cmp(src, dst, shallow=False):
                continue
            drift.append(dst)
            if not args.check:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"synced  {dst.relative_to(REPO_ROOT)}")

    if not drift:
        return 0

    if args.check:
        sys.stderr.write(
            f"\nDrift detected in {len(drift)} file(s). Edit the canonical copy under "
            f"skills/{CANONICAL}/ and run `python skills/sync_shared.py` to propagate:\n"
        )
        for path in drift:
            sys.stderr.write(f"  - {path.relative_to(REPO_ROOT)}\n")
        return 1

    # Sync mode: files have been written. Exit 0 — pre-commit detects the file
    # modifications via its own diff and will fail the run regardless.
    print(f"\nSynced {len(drift)} file(s) from skills/{CANONICAL}/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
