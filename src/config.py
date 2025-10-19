from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any

from src.utils.config_utils import (
    resolve_config_path,
    load_yaml_config,
    deep_update,
)


@dataclass
class FeatureWindows:
    vol_1s: int = 1
    delta_1s: int = 1
    ma: List[int] = field(default_factory=lambda: [7, 15, 30])


@dataclass
class SignalThresholds:
    imbalance: float = 0.6
    max_spread_bps: float = 1.5
    # Mean-reversion parameters
    mr_min_revert_bps: float = 2.0
    mr_expected_bps: float = 6.0
    mr_conf_norm_bps: float = 5.0
    mr_max_imbalance: float = 1.0


@dataclass
class Horizons:
    scalp: int = 30
    ttl_s: int = 10


@dataclass
class Streams:
    aggTrade: bool = True
    depth_100ms: bool = True
    kline_1s: bool = True


@dataclass
class Rules:
    momentum_enabled: bool = True
    mean_reversion_enabled: bool = True


@dataclass
class UIConfig:
    auto_refresh_seconds: int = 2
    show_debug: bool = False


@dataclass
class AppConfig:
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    streams: Streams = field(default_factory=Streams)
    features: FeatureWindows = field(default_factory=FeatureWindows)
    signal_thresholds: SignalThresholds = field(default_factory=SignalThresholds)
    horizons: Horizons = field(default_factory=Horizons)
    storage: Dict[str, Any] = field(
        default_factory=lambda: {"logbook_dir": "data/logbook"}
    )
    ui: UIConfig = field(default_factory=UIConfig)
    rules: Rules = field(default_factory=Rules)


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    # Backward-compat shim to avoid breaking imports; delegate to utils
    return deep_update(base, override)


def load_app_config(config_path: str | None = None) -> AppConfig:
    cfg_path = config_path or resolve_config_path()
    file_cfg = load_yaml_config(cfg_path) if os.path.exists(cfg_path) else {}

    # Environment overrides
    env_overrides: Dict[str, Any] = {}
    logbook_dir = os.getenv("LOGBOOK_DIR")
    if logbook_dir:
        env_overrides.setdefault("storage", {})["logbook_dir"] = logbook_dir

    merged = deep_update(file_cfg, env_overrides) if env_overrides else file_cfg

    # Build dataclasses
    return AppConfig(
        symbols=merged.get("symbols", ["BTCUSDT"]),
        streams=Streams(**merged.get("streams", {})),
        features=FeatureWindows(**merged.get("features", {})),
        signal_thresholds=SignalThresholds(**merged.get("signal_thresholds", {})),
        horizons=Horizons(**merged.get("horizons", {})),
        storage=merged.get("storage", {"logbook_dir": "data/logbook"}),
        ui=UIConfig(**merged.get("ui", {})),
        rules=Rules(**merged.get("rules", {})),
    )


__all__ = [
    "AppConfig",
    "load_app_config",
]
