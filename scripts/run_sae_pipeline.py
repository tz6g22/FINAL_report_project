#!/usr/bin/env python3
"""Wrapper script for the dual-model SAE pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stream_analysis.sae.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
