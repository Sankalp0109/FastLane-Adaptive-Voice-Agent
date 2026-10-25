"""Resolve model roles ("large", "small", ...) to clients."""

from __future__ import annotations

import os
from functools import lru_cache

from fastlane.config import MODEL_ROLES
from fastlane.models.base import LLMClient


def role_spec(role: str) -> str:
    return os.environ.get(f"FASTLANE_{role.upper()}", MODEL_ROLES.get(role, role))


@lru_cache(maxsize=16)
def client_for(spec: str) -> LLMClient:
    """spec is "provider:model" (e.g. "groq:openai/gpt-oss-120b") or a role name.

    The "large" role gets the "large_backup" model as a rate-limit fallback unless
    FASTLANE_LARGE_BACKUP=none."""
    from fastlane.models.openai_compat import FallbackClient, OpenAICompatClient

    role = spec if (spec in MODEL_ROLES or os.environ.get(f"FASTLANE_{spec.upper()}")) else None
    if role:
        spec = role_spec(role)
    provider, _, model = spec.partition(":")
    client = OpenAICompatClient(provider, model)
    if role == "large":
        backup = role_spec("large_backup")
        if backup and backup.lower() != "none":
            b_provider, _, b_model = backup.partition(":")
            return FallbackClient(client, OpenAICompatClient(b_provider, b_model))
    return client


def ollama_has(model: str) -> bool:
    """True if a local Ollama server has `model` pulled (used to fall back before LoRA exists)."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            names = {m["name"] for m in json.load(r).get("models", [])}
    except OSError:
        return False
    return model in names or f"{model}:latest" in names


def small_model_spec(kind: str) -> str:
    """kind: 'lora' | 'lora_mb' | 'prompted'. Falls back to the untuned base when a model is not available
    (adapter not built yet, or no MLX server running)."""
    from fastlane.config import PROJECT_DIR

    role = {"lora": "small_lora", "lora_mb": "small_lora_mb", "prompted": "small"}[kind]
    spec = role_spec(role)
    if spec.startswith("mlx:"):
        path = PROJECT_DIR / spec.split(":", 1)[1]
        if not path.exists() or not _server_up("http://localhost:8080/v1/models"):
            if kind != "prompted":
                return small_model_spec("prompted")
            return role_spec("small_ollama")
    if kind != "prompted" and spec.startswith("ollama:") and not ollama_has(spec.split(":", 1)[1]):
        return small_model_spec("prompted")
    return spec


def _server_up(url: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=2):
            return True
    except OSError:
        return False


def small_client(cfg) -> tuple[LLMClient | None, bool]:
    """(client, compact_prompt) for a system's small model. The compact prompt is used only when the
    fine-tuned model is actually available; before training, the base model gets the full prompt."""
    if cfg.small_model == "none":
        return None, False
    kind = "prompted" if cfg.small_model == "prompted" else ("lora_mb" if cfg.adapter == "movies_buses" else "lora")
    spec = small_model_spec(kind)
    tuned = kind != "prompted" and spec not in (role_spec("small"), role_spec("small_ollama"))
    return client_for(spec), tuned
