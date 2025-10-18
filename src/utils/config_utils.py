from __future__ import annotations

import os
from typing import Any, Dict

from dotenv import load_dotenv
import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def resolve_config_path(
    env_var: str = "CONFIG_PATH", default_path: str | None = None
) -> str:
    load_dotenv(override=False)
    if default_path is None:
        default_path = os.path.join("configs", "config.yaml")
    return os.getenv(env_var, default_path)


__all__ = ["load_yaml_config", "deep_update", "resolve_config_path"]
