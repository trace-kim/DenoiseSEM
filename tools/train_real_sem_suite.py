"""Run the five agreed real-SEM pipelines sequentially, continuing after failure."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_denoise.real_suite import main


if __name__ == "__main__":
    raise SystemExit(main())
