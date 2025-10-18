from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


DEFAULT_CONTROL_DIR = os.getenv("CONTROL_DIR", os.path.join("data", "control"))


@dataclass
class RuntimeConfigPaths:
    base_dir: str = DEFAULT_CONTROL_DIR

    @property
    def runtime_file(self) -> str:
        return os.path.join(self.base_dir, "runtime_config.json")


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
        "rules": {"momentum_enabled": bool, "mean_reversion_enabled": bool},
        "signal_thresholds": {"imbalance": float, "max_spread_bps": float},
        "horizons": {"scalp": int, "ttl_s": int}
      }
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.paths = RuntimeConfigPaths(base_dir or DEFAULT_CONTROL_DIR)
        self._last_loaded: Optional[Dict[str, Any]] = None

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
        # Signal engine thresholds/rules/horizons
        if signal_engine is not None:
            if isinstance(overrides.get("signal_thresholds"), dict):
                signal_engine.thr = _merge(signal_engine.thr, overrides["signal_thresholds"])  # type: ignore[attr-defined]
            if isinstance(overrides.get("rules"), dict):
                signal_engine.rules = _merge(signal_engine.rules, overrides["rules"])  # type: ignore[attr-defined]
            if isinstance(overrides.get("horizons"), dict):
                signal_engine.hz = _merge(signal_engine.hz, overrides["horizons"])  # type: ignore[attr-defined]

        # Evaluator horizon (scalp)
        if evaluator is not None and isinstance(overrides.get("horizons"), dict):
            scalp = overrides["horizons"].get("scalp")
            if isinstance(scalp, int) and scalp > 0:
                try:
                    evaluator.horizon_ms = int(scalp) * 1000
                except Exception:
                    pass


__all__ = ["RuntimeConfigManager"]
