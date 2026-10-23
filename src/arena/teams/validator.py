"""Team legality validation via the local Showdown server's CLI.

`pokemon-showdown validate-team <format>` reads a team from stdin, exits 0 if
legal, else 1 with reasons on stdout/stderr. We convert Showdown paste -> packed
via poke-env before piping. Used to prune stale sample teams (metagame banlists
drift) and, in v2, to validate agent-built teams.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from poke_env.teambuilder.teambuilder import Teambuilder

_DEFAULT_PS_DIR = Path(__file__).resolve().parents[3].parent / "pokemon-showdown"


@dataclass
class ValidationResult:
    ok: bool
    reason: str


def to_packed(team: str) -> str:
    """Accept paste or packed; return packed."""
    stripped = team.strip()
    # packed teams are single-line, pipe-delimited, no newlines between mons
    if "\n" not in stripped and "|" in stripped:
        return stripped
    return Teambuilder.join_team(Teambuilder.parse_showdown_team(team))


def validate(team: str, fmt: str, ps_dir: Optional[Path] = None, timeout: float = 30.0) -> ValidationResult:
    ps_dir = Path(ps_dir or _DEFAULT_PS_DIR)
    packed = to_packed(team)
    try:
        proc = subprocess.run(
            ["node", "pokemon-showdown", "validate-team", fmt],
            cwd=str(ps_dir),
            input=packed,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return ValidationResult(False, f"node/pokemon-showdown not found in {ps_dir}")
    except subprocess.TimeoutExpired:
        return ValidationResult(False, "validator timeout")
    reason = (proc.stdout + proc.stderr).strip()
    return ValidationResult(proc.returncode == 0, reason)
