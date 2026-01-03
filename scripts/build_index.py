# Wrapper for convenience: keeps backwards compatible path scripts/build_index.py
from pathlib import Path
import sys

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_index import main

if __name__ == "__main__":
    main()
