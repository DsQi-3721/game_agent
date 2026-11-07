"""Model registry: name -> ModelSpec (provider/litellm-id/base_url/key/flags).

Loaded from configs/models/registry.yaml. API keys resolve from env vars at use
time (a project-root .env is loaded if present), so secrets never live in YAML.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_REGISTRY = _PROJECT_ROOT / "configs" / "models" / "registry.yaml"
_ENV_FILE = _PROJECT_ROOT / ".env"


def _load_dotenv(path: Path = _ENV_FILE) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_vars(value):
    """``${VAR}`` / ``${VAR:-default}`` in registry values: machine-specific ports and paths
    (e.g. ``api_base: ${ARENA_VLLM_BASE:-http://localhost:8001/v1}``) without editing the YAML."""
    if isinstance(value, str):
        return _VAR.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: expand_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_vars(v) for v in value]
    return value


@dataclass
class ModelSpec:
    name: str
    litellm_id: str
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None
    use_proxy: bool = False
    supports_tools: bool = True
    price_in: float = 0.0   # USD per 1M input tokens (authoritative when > 0)
    price_out: float = 0.0  # USD per 1M output tokens
    notes: str = ""
    # --- optional serving knobs (local OpenAI-compatible servers: mlx_lm / vLLM) ---
    extra_body: dict = field(default_factory=dict)  # merged into every request body
                                                    # (e.g. chat_template_kwargs, top_k)
    context_window: int = 0                 # served max context in tokens (0 = unknown)
    max_tokens: Optional[int] = None        # default completion cap if the caller sets none
    timeout_s: Optional[float] = None       # per-request timeout
    max_retries: Optional[int] = None       # None -> litellm default; 0 for local servers
    tokenizer_path: Optional[str] = None    # local HF tokenizer dir (exact token counts)

    def cost(self, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        """Cost from the local price table, or None to defer to litellm."""
        if self.price_in or self.price_out:
            return (prompt_tokens * self.price_in + completion_tokens * self.price_out) / 1e6
        return None

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    def require_key(self) -> str:
        key = self.api_key
        if not key:
            raise RuntimeError(
                f"model {self.name!r} needs env var {self.api_key_env}; "
                f"set it or add it to {_ENV_FILE}"
            )
        return key


class ModelRegistry:
    def __init__(self, specs: dict[str, ModelSpec]):
        self._specs = specs

    @classmethod
    def load(cls, path: Path | str = _DEFAULT_REGISTRY) -> "ModelRegistry":
        _load_dotenv()
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        specs = {}
        for name, cfg in (data.get("models") or {}).items():
            specs[name] = ModelSpec(name=name, **expand_vars(cfg))
        return cls(specs)

    def get(self, name: str) -> ModelSpec:
        if name not in self._specs:
            raise KeyError(f"unknown model {name!r}; have {sorted(self._specs)}")
        return self._specs[name]

    def names(self) -> list[str]:
        return sorted(self._specs)
