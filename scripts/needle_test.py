#!/usr/bin/env python3
"""Checkout launcher for `friday needle`; implementation ships in the package."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from friday.commands.needle import main

    raise SystemExit(main())
