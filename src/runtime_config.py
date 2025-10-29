from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


DEFAULT_CONTROL_DIR = os.path.join("data", "control")


@dataclass
class RuntimeConfigPaths:
    base_dir: str = ""

    def __post_init__(self) -> None:
        # Resolve at instantiation time to honor current environment
        if not self.base_dir:
            self.base_dir = os.getenv("CONTROL_DIR", DEFAULT_CONTROL_DIR)

    @property
    def runtime_file(self) -> str:
        return os.path.join(self.base_dir, "runtime_config.json")

    @property
    def llm_configs_file(self) -> str:
        return os.path.join(self.base_dir, "llm_configs.json")


def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_write_json(path: str, data: Dict[str, Any]) -> bool:
    tmp_path = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class RuntimeConfigManager:
    """Best-effort runtime override manager stored under CONTROL_DIR.

    Schema (partial, optional keys):
      {
        "llm": {...},
        "symbols": ["BTCUSDT", ...],
        "trader": {
            "concurrent_positions": 1,
            # global confidence_threshold removed; use per-direction thresholds in runtime_config via UI
            "default_position_size_usd": 0.0,
            "default_leverage": null,
            "tp_percent": 0.0,
            "sl_percent": 0.0,
            "trailing_sl_enabled": false,
            "tp_disabled": false,
            "auto_expire_minutes": null
        },
        "horizons": {"scalp": int, "ttl_s": int}
      }
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        resolved_base = base_dir or os.getenv("CONTROL_DIR", DEFAULT_CONTROL_DIR)
        self.paths = RuntimeConfigPaths(resolved_base)
        self._last_loaded: Optional[Dict[str, Any]] = None
        self._last_llm_mtime: Optional[float] = None

    def read(self) -> Optional[Dict[str, Any]]:
        return _safe_read_json(self.paths.runtime_file)

    def write(self, data: Dict[str, Any]) -> bool:
        return _safe_write_json(self.paths.runtime_file, data)

    def load_if_changed(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        cfg = self.read()
        if cfg is None:
            if self._last_loaded is None:
                return (False, None)
            # File disappeared; treat as no change
            return (False, None)
        if cfg != self._last_loaded:
            self._last_loaded = cfg
            return (True, cfg)
        return (False, cfg)

    # ---------- LLM configs: separate file llm_configs.json ----------
    def _llm_file_mtime(self) -> Optional[float]:
        try:
            return os.path.getmtime(self.paths.llm_configs_file)
        except Exception:
            return None

    def read_llm_configs(self) -> Optional[Dict[str, Any]]:
        return _safe_read_json(self.paths.llm_configs_file)

    def load_llm_configs_if_changed(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        mtime = self._llm_file_mtime()
        if mtime is None:
            if self._last_llm_mtime is None:
                return (False, None)
            return (False, None)
        if self._last_llm_mtime is None or mtime != self._last_llm_mtime:
            self._last_llm_mtime = mtime
            return (True, self.read_llm_configs())
        return (False, self.read_llm_configs())

    @staticmethod
    def get_active_llm_config_from(
        configs_doc: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        try:
            configs = (configs_doc or {}).get("configs", [])
            if not isinstance(configs, list):
                return None
            for cfg in configs:
                if isinstance(cfg, dict) and cfg.get("is_active"):
                    return cfg
            return configs[0] if configs else None
        except Exception:
            return None

    # ---------- Application helpers ----------
    @staticmethod
    def apply_to_engines(
        overrides: Dict[str, Any] | None,
        *,
        signal_engine: Any | None = None,
        evaluator: Any | None = None,
    ) -> None:
        """Apply runtime overrides to live components in-place.

        Components are optional; only provided ones are updated.
        """
        if not overrides:
            return
        # Currently a placeholder for future live updates to engines
        return


__all__ = ["RuntimeConfigManager"]
