"""Read API keys from the project's .env file (never printed or logged)."""

from __future__ import annotations

import os
from functools import lru_cache

from fastlane.config import PROJECT_DIR


@lru_cache(maxsize=1)
def _dotenv() -> dict[str, str]:
    path = PROJECT_DIR / ".env"
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.removeprefix("export ").strip()] = value.strip().strip("'\"")
    return values


def get_key(name: str) -> str:
    """Environment variable first, then .env. Raises with a hint if missing."""
    value = os.environ.get(name) or _dotenv().get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; add it to .env or your shell")
    return value
