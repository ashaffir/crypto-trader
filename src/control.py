from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


DEFAULT_CONTROL_DIR = os.getenv("CONTROL_DIR", os.path.join("data", "control"))


@dataclass
class ControlPaths:
    base_dir: str = DEFAULT_CONTROL_DIR

    @property
    def control_file(self) -> str:
        return os.path.join(self.base_dir, "bot_control.json")

    @property
    def status_file(self) -> str:
        return os.path.join(self.base_dir, "bot_status.json")


class Control:
    """Filesystem-based control and status channel shared between bot and UI.

    - Desired state is stored in `bot_control.json` as {"desired": "running"|"stopped"}.
    - Runtime status/heartbeat is stored in `bot_status.json` with last activity.
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.paths = ControlPaths(base_dir or DEFAULT_CONTROL_DIR)
        os.makedirs(self.paths.base_dir, exist_ok=True)

    # ----- Desired state API -----
    def set_desired_state(self, desired: str) -> None:
        desired_norm = "running" if desired == "running" else "stopped"
        data = {"desired": desired_norm, "updated_ts": int(time.time() * 1000)}
        self._safe_write(self.paths.control_file, data)

    def get_desired_state(self) -> str:
        data = self._safe_read(self.paths.control_file)
        if not isinstance(data, dict):
            return "stopped"
        desired = str(data.get("desired") or "stopped").lower()
        return "running" if desired == "running" else "stopped"

    # ----- Status/heartbeat API -----
    def write_status(self, status: Dict[str, Any]) -> None:
        payload = {
            **status,
            "heartbeat_ts": int(time.time() * 1000),
        }
        self._safe_write(self.paths.status_file, payload)

    def read_status(self) -> Dict[str, Any]:
        data = self._safe_read(self.paths.status_file)
        return data if isinstance(data, dict) else {}

    # ----- Internals -----
    @staticmethod
    def _safe_read(path: str) -> Any:
        try:
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _safe_write(path: str, data: Dict[str, Any]) -> None:
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass


__all__ = ["Control", "ControlPaths"]
