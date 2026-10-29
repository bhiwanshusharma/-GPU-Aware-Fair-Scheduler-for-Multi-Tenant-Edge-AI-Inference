"""CLI entry point for launching the GPU Scheduler Monitoring Dashboard.

Usage:
    python -m simulator.dashboard --port 8501
    python -m simulator.dashboard --headless
"""

import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_APP = PROJECT_ROOT / "src" / "dashboard" / "app.py"


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs command line argument parser for dashboard launcher."""
    parser = argparse.ArgumentParser(
        prog="python -m simulator.dashboard",
        description="Launch Lightweight Monitoring Dashboard for GPU-Aware Fair Scheduler",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port to serve the dashboard on (default: 8501)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address to bind to (default: localhost)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without automatically launching a browser window",
    )
    return parser


def run_dashboard(args: argparse.Namespace) -> int:
    """Launches the Streamlit monitoring dashboard server."""
    if not DASHBOARD_APP.exists():
        print(f"Error: Dashboard script not found at {DASHBOARD_APP}")
        return 1

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(DASHBOARD_APP),
        "--server.port", str(args.port),
        "--server.address", str(args.host),
    ]
    if args.headless:
        cmd.extend(["--server.headless", "true"])

    print("=" * 80)
    print("  LAUNCHING GPU-AWARE FAIR SCHEDULER MONITORING DASHBOARD")
    print("=" * 80)
    print(f"App Path : {DASHBOARD_APP}")
    print(f"URL      : http://{args.host}:{args.port}")
    print("=" * 80)

    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        print("\n[Dashboard] Server stopped by user.")
        return 0


def main() -> None:
    """CLI execution entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    sys.exit(run_dashboard(args))


if __name__ == "__main__":
    main()
