#!/usr/bin/env python
"""Export n-best span candidates from an official RECCON best_model checkpoint."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    argv = sys.argv[1:]
    if "--backend" not in argv:
        argv.extend(["--backend", "hf_qa"])
    if "--condition" not in argv:
        argv.extend(["--condition", "official_candidate_baseline"])
    sys.argv = [str(Path(__file__).with_name("run_reccon_baseline.py")), *argv]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
