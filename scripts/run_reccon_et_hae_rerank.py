#!/usr/bin/env python
"""Convenience wrapper for ET-HAE RECCON reranking."""

from __future__ import annotations

import sys
import runpy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


if __name__ == "__main__":
    sys.argv.extend(["--condition", "et_hae"])
    runpy.run_path(str(Path(__file__).with_name("run_reccon_rerank.py")), run_name="__main__")
