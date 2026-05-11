#!/usr/bin/env python3
#file: ULDAS.py
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated.*",
    category=UserWarning,
    module="ctranslate2.*",
)

import sys
import os

# Ensure the project root is on sys.path so `uldas` package is importable
# regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uldas.main import main

if __name__ == "__main__":
    main()