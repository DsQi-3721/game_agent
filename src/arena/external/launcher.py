"""Spawn + supervise an external opponent process.

Lifecycle: ``start()`` spawns the subprocess (or ``docker run``), streaming its
stdout/stderr to ``run_dir/external_<name>.log``; ``wait_ready()`` blocks until a
line matches ``ready_regex`` (the bot logged in and is accepting challenges) or
times out; ``is_alive()`` is polled by the match loop so a dead bot doesn't hang
``send_challenges`` forever; ``stop()`` terminates the whole process group.

We pass the handshake parameters (server/username/opponent/format/n) via env vars
so the external ``run.sh`` shim can map them onto its own evaluate entrypoint.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
from pathlib import Path
from typing import Optional

from arena.config.schema import ExternalConfig


class ExternalLaunchError(RuntimeError):
    pass


class ExternalLauncher:
    def __init__(self, cfg: ExternalConfig, name: str):
        self.cfg = cfg
        self.name = name
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._log_path: Optional[Path] = None
        self._ready = re.compile(cfg.ready_regex, re.IGNORECASE)
        self._tail_task: Optional[asyncio.Task] = None
        self._ready_evt = asyncio.Event()

    # --- lifecycle -------------------------------------------------------
    async def start(self, fmt: str, opponent_username: str, run_dir: Path,
                    username: Optional[str] = None) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = run_dir / f"external_{self.name}.log"

        env = dict(os.environ)
        env.update(self.cfg.env)
        env.update({
            "ARENA_SERVER": "ws://localhost:8000/showdown/websocket",
            "ARENA_USERNAME": username or self.cfg.username,  # fresh name per launch
            "ARENA_OPPONENT": opponent_username,
            "ARENA_FORMAT": fmt,
        })

        if self.cfg.docker_image:
            argv = ["docker", "run", "--rm", "--network", "host"]
            for k, v in env.items():
                if k.startswith("ARENA_"):
                    argv += ["-e", f"{k}={v}"]
            argv += [self.cfg.docker_image, *self.cfg.launch_cmd]
        else:
            argv = list(self.cfg.launch_cmd)

        self._logf = open(self._log_path, "wb")  # noqa: SIM115 (closed in stop())
        self._proc = await asyncio.create_subprocess_exec(
            *argv, cwd=self.cfg.cwd, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,  # own process group -> clean group kill
        )
        self._tail_task = asyncio.create_task(self._pump_stdout())

    async def _pump_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        async for raw in self._proc.stdout:
            self._logf.write(raw)
            self._logf.flush()
            if not self._ready_evt.is_set() and self._ready.search(raw.decode("utf-8", "replace")):
                self._ready_evt.set()

    async def wait_ready(self) -> None:
        try:
            await asyncio.wait_for(self._ready_evt.wait(), timeout=self.cfg.ready_timeout_s)
        except asyncio.TimeoutError as e:
            await self.stop()
            raise ExternalLaunchError(
                f"external {self.name!r} not ready within {self.cfg.ready_timeout_s}s "
                f"(see {self._log_path})"
            ) from e

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def stop(self) -> None:
        if self._tail_task is not None:
            self._tail_task.cancel()
        if self._proc is not None and self._proc.returncode is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if getattr(self, "_logf", None) is not None:
            try:
                self._logf.close()
            except Exception:  # noqa: BLE001
                pass
