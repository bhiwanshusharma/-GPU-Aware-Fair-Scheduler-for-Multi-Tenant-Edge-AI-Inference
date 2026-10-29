"""Alias for simulator.plot_results allowing execution via src.visualization.plot_results."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.plot_results import main, run_plotting, build_arg_parser

if __name__ == "__main__":
    main()
