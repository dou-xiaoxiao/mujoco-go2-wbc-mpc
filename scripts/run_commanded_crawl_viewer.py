"""Clean entry point for the command-driven crawl viewer demo."""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from run_srb_mpc_crawl_continuous_viewer import main  # noqa: E402


if __name__ == "__main__":
    main()
