"""Build the argv that launches an evaluation as a detached subprocess.

The TUI runs the SAME engine as the CLI (`arena run ... --emit-events`), so config
is the single source of truth and CLI/TUI stay in lockstep.
"""

from __future__ import annotations

import sys
from pathlib import Path


def build_argv(config_path: Path | str, run_dir: Path | str) -> list[str]:
    return [
        sys.executable, "-m", "arena.cli", "run", str(config_path),
        "--emit-events", "--run-dir", str(run_dir),
    ]
