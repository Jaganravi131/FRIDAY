#!/usr/bin/env python3
"""Checkout launcher for `friday ablation`; implementation ships in the package."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from friday.commands.ablation import main

    raise SystemExit(main())
