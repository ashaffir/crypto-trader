import json
import os
from typing import Optional

from .common import CONTROL_DIR


_DEFAULTS: dict[str, object] = {
    "logical_max_files": 10,
    "quality_max_files": None,  # None means "all"
    "chart_points": 200,
}
# ---- Trader settings ----


def load_trader_settings(base_dir: Optional[str] = None) -> dict:
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    raw = cfg.get("trader") if isinstance(cfg, dict) else None
    out: dict[str, object] = {
        "concurrent_positions": 1,
        "confidence_threshold": 0.8,
        "default_position_size_usd": 0.0,
        "default_leverage": None,
        "tp_percent": 0.0,
        "sl_percent": 0.0,
        "trailing_sl_enabled": False,
        "tp_disabled": False,
        "auto_expire_minutes": None,
    }
    if isinstance(raw, dict):
        for k in list(out.keys()):
            if k in raw:
                out[k] = raw[k]
    return out


def save_trader_settings(settings: dict, base_dir: Optional[str] = None) -> bool:
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    merged = dict(cfg or {})
    cur = load_trader_settings(base_dir)
    for k, v in settings.items():
        cur[k] = v
    merged["trader"] = cur
    return _safe_write_json(path, merged)


# ---- Trader fee settings ----


def load_trader_fee_settings(base_dir: Optional[str] = None) -> dict:
    """Load trader fee settings nested under trader.fees with sensible defaults.

    Defaults:
      enabled=False, market="spot", vip_tier=0, liquidity="taker", bnb_discount=False
    """
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    trader = cfg.get("trader") if isinstance(cfg, dict) else None
    fees = trader.get("fees") if isinstance(trader, dict) else None
    out: dict[str, object] = {
        "enabled": False,
        "market": "spot",
        "vip_tier": 0,
        "liquidity": "taker",
        "bnb_discount": False,
    }
    if isinstance(fees, dict):
        if "enabled" in fees:
            out["enabled"] = bool(fees.get("enabled", False))
        if str(fees.get("market", "spot")).lower() in ("spot", "futures"):
            out["market"] = str(fees.get("market")).lower()
        try:
            if fees.get("vip_tier") not in (None, ""):
                out["vip_tier"] = max(0, min(9, int(fees.get("vip_tier"))))
        except Exception:
            pass
        if str(fees.get("liquidity", "taker")).lower() in ("maker", "taker"):
            out["liquidity"] = str(fees.get("liquidity")).lower()
        if "bnb_discount" in fees:
            out["bnb_discount"] = bool(fees.get("bnb_discount", False))
    return out


def save_trader_fee_settings(settings: dict, base_dir: Optional[str] = None) -> bool:
    """Persist trader fee settings under trader.fees, merging with existing.

    Only known keys are applied; others are ignored.
    """
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    merged = dict(cfg or {})
    trader = dict((merged.get("trader") or {}))
    current = load_trader_fee_settings(base_dir)

    keep: dict[str, object] = dict(current)
    if isinstance(settings, dict):
        if "enabled" in settings:
            keep["enabled"] = bool(settings.get("enabled"))
        if str(settings.get("market", keep.get("market", "spot"))).lower() in (
            "spot",
            "futures",
        ):
            keep["market"] = str(
                settings.get("market", keep.get("market", "spot"))
            ).lower()
        try:
            if settings.get("vip_tier") not in (None, ""):
                keep["vip_tier"] = max(0, min(9, int(settings.get("vip_tier"))))
        except Exception:
            pass
        if str(settings.get("liquidity", keep.get("liquidity", "taker"))).lower() in (
            "maker",
            "taker",
        ):
            keep["liquidity"] = str(
                settings.get("liquidity", keep.get("liquidity", "taker"))
            ).lower()
        if "bnb_discount" in settings:
            keep["bnb_discount"] = bool(settings.get("bnb_discount"))

    trader["fees"] = keep
    merged["trader"] = trader
    return _safe_write_json(path, merged)


def _runtime_file(base_dir: Optional[str] = None) -> str:
    b = base_dir or CONTROL_DIR
    return os.path.join(b, "runtime_config.json")


def _safe_read_json(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except Exception:
        return {}


def _safe_write_json(path: str, data: dict) -> bool:
    tmp = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


def load_backtesting_settings(base_dir: Optional[str] = None) -> dict:
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    raw = cfg.get("backtesting") if isinstance(cfg, dict) else None
    out: dict[str, object] = dict(_DEFAULTS)
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k in out:
                out[k] = v
    return out


def save_backtesting_settings(settings: dict, base_dir: Optional[str] = None) -> bool:
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    keep: dict[str, object] = {}
    for k in _DEFAULTS.keys():
        if k in settings:
            keep[k] = settings[k]
    merged = dict(cfg or {})
    merged["backtesting"] = dict(load_backtesting_settings(base_dir))
    merged["backtesting"].update(keep)
    return _safe_write_json(path, merged)


# ---- Tracked symbols and LLM settings ----


def load_tracked_symbols(base_dir: Optional[str] = None) -> list[str]:
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    raw = cfg.get("symbols") if isinstance(cfg, dict) else None
    if isinstance(raw, list):
        return [str(x).upper() for x in raw if isinstance(x, (str,))]
    # Fallback to config default
    return ["BTCUSDT"]


def save_tracked_symbols(symbols: list[str], base_dir: Optional[str] = None) -> bool:
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    merged = dict(cfg or {})
    merged["symbols"] = [str(s).upper() for s in symbols if s]
    return _safe_write_json(path, merged)


def load_llm_settings(base_dir: Optional[str] = None) -> dict:
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    raw = cfg.get("llm") if isinstance(cfg, dict) else None
    # Defaults
    out: dict[str, object] = {
        "active": "default",
        "window_seconds": 60,
        "refresh_seconds": 5,
        "debug_save_request": False,
        "configs": {
            "default": {
                "base_url": "",
                "api_key": "",
                "model": "",
                "system_prompt": "",
                "user_template": "",
            }
        },
    }
    if isinstance(raw, dict):
        # shallow merge for known keys
        for k in ("active", "window_seconds", "refresh_seconds", "debug_save_request"):
            if k in raw:
                out[k] = raw[k]
        cfgs = raw.get("configs")
        if isinstance(cfgs, dict) and cfgs:
            out["configs"] = cfgs
    return out


def save_llm_settings(settings: dict, base_dir: Optional[str] = None) -> bool:
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    merged = dict(cfg or {})
    keep: dict[str, object] = {}
    for k in (
        "active",
        "window_seconds",
        "refresh_seconds",
        "debug_save_request",
        "configs",
    ):
        if k in settings:
            keep[k] = settings[k]
    cur = load_llm_settings(base_dir)
    cur.update(keep)
    merged["llm"] = cur
    return _safe_write_json(path, merged)


def _llm_configs_file(base_dir: Optional[str] = None) -> str:
    b = base_dir or CONTROL_DIR
    return os.path.join(b, "llm_configs.json")


def load_llm_configs(base_dir: Optional[str] = None) -> list[dict]:
    """Load all LLM configurations from llm_configs.json"""
    path = _llm_configs_file(base_dir)
    data = _safe_read_json(path)
    configs = data.get("configs", [])
    if isinstance(configs, list):
        return configs
    return []


def save_llm_configs(configs: list[dict], base_dir: Optional[str] = None) -> bool:
    """Save all LLM configurations to llm_configs.json"""
    path = _llm_configs_file(base_dir)
    data = {"configs": configs}
    return _safe_write_json(path, data)


def get_active_llm_config(base_dir: Optional[str] = None) -> Optional[dict]:
    """Get the active LLM configuration"""
    configs = load_llm_configs(base_dir)
    for cfg in configs:
        if cfg.get("is_active"):
            return cfg
    return configs[0] if configs else None


def set_active_llm(name: str, base_dir: Optional[str] = None) -> bool:
    """Set the specified LLM as active"""
    configs = load_llm_configs(base_dir)
    found = False
    for cfg in configs:
        if cfg.get("name") == name:
            cfg["is_active"] = True
            found = True
        else:
            cfg["is_active"] = False
    if found:
        return save_llm_configs(configs, base_dir)
    return False


def delete_llm_config(name: str, base_dir: Optional[str] = None) -> bool:
    """Delete an LLM configuration by name"""
    configs = load_llm_configs(base_dir)
    new_configs = [cfg for cfg in configs if cfg.get("name") != name]
    if len(new_configs) < len(configs):
        return save_llm_configs(new_configs, base_dir)
    return False


def upsert_llm_config(config: dict, base_dir: Optional[str] = None) -> bool:
    """Create or update an LLM configuration"""
    configs = load_llm_configs(base_dir)
    name = config.get("name")
    if not name:
        return False

    # Update existing or append new
    found = False
    for i, cfg in enumerate(configs):
        if cfg.get("name") == name:
            configs[i] = config
            found = True
            break

    if not found:
        configs.append(config)

    return save_llm_configs(configs, base_dir)


def load_window_seconds(base_dir: Optional[str] = None) -> int:
    """Load the window size in seconds for LLM data analysis"""
    settings = load_llm_settings(base_dir)
    window = settings.get("window_seconds", 60)
    return int(window) if isinstance(window, (int, float)) else 60


def save_window_seconds(seconds: int, base_dir: Optional[str] = None) -> bool:
    """Save the window size in seconds for LLM data analysis"""
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    merged = dict(cfg or {})

    # Load current LLM settings and update only window_seconds
    llm_settings = load_llm_settings(base_dir)
    llm_settings["window_seconds"] = int(seconds)
    merged["llm"] = llm_settings

    return _safe_write_json(path, merged)


def load_sidebar_settings(base_dir: Optional[str] = None) -> dict:
    """Load sidebar settings; default symbol is first tracked; refresh=2."""
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    sidebar = cfg.get("sidebar") if isinstance(cfg, dict) else None
    tracked = load_tracked_symbols(base_dir)
    default_symbol = tracked[0] if tracked else "BTCUSDT"
    default_refresh = 2
    out = {"symbol": default_symbol, "refresh_seconds": default_refresh}
    if isinstance(sidebar, dict):
        sym = sidebar.get("symbol")
        if isinstance(sym, str) and sym.upper() in set(tracked or []):
            out["symbol"] = sym.upper()
        try:
            rs = int(sidebar.get("refresh_seconds", default_refresh))
            if rs >= 1:
                out["refresh_seconds"] = rs
        except Exception:
            pass
    return out


def save_sidebar_settings(settings: dict, base_dir: Optional[str] = None) -> bool:
    """Persist sidebar settings (symbol, refresh_seconds)."""
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    merged = dict(cfg or {})
    keep: dict[str, object] = {}
    if isinstance(settings, dict):
        if isinstance(settings.get("symbol"), str):
            keep["symbol"] = settings["symbol"].upper()
        if "refresh_seconds" in settings:
            try:
                keep["refresh_seconds"] = max(1, int(settings["refresh_seconds"]))
            except Exception:
                pass
    cur = load_sidebar_settings(base_dir)
    cur.update(keep)
    merged["sidebar"] = cur
    return _safe_write_json(path, merged)


__all__ = [
    "load_backtesting_settings",
    "save_backtesting_settings",
    "load_tracked_symbols",
    "save_tracked_symbols",
    "load_llm_settings",
    "save_llm_settings",
    "load_llm_configs",
    "save_llm_configs",
    "get_active_llm_config",
    "set_active_llm",
    "delete_llm_config",
    "upsert_llm_config",
    "load_window_seconds",
    "save_window_seconds",
    # Sidebar helpers
    "load_sidebar_settings",
    "save_sidebar_settings",
]

# ---- Market mode (spot/futures) helpers ----


def load_market_mode(base_dir: Optional[str] = None) -> str:
    """Load current market mode from runtime_config.json top-level key 'market'.

    Defaults to 'spot' when missing/invalid.
    """
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    val = str((cfg or {}).get("market", "spot")).lower()
    return val if val in ("spot", "futures") else "spot"


def save_market_mode(mode: str, base_dir: Optional[str] = None) -> bool:
    """Persist market mode ('spot' | 'futures') under top-level 'market'."""
    m = str(mode).lower()
    if m not in ("spot", "futures"):
        return False
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    merged = dict(cfg or {})
    merged["market"] = m
    return _safe_write_json(path, merged)


# ---- Execution settings (mode/venue/network/keys) ----


def load_execution_settings(base_dir: Optional[str] = None) -> dict:
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    ex = cfg.get("execution") if isinstance(cfg, dict) else None
    out = {
        "mode": "paper",
        "venue": "spot",
        "network": "testnet",
        "api_key": None,
        "api_secret": None,
    }
    if isinstance(ex, dict):
        for k in out.keys():
            if k in ex:
                out[k] = ex[k]
    return out


def save_execution_settings(settings: dict, base_dir: Optional[str] = None) -> bool:
    path = _runtime_file(base_dir)
    cfg = _safe_read_json(path)
    merged = dict(cfg or {})
    cur = load_execution_settings(base_dir)
    for k in cur.keys():
        if k in settings:
            cur[k] = settings[k]
    merged["execution"] = cur
    return _safe_write_json(path, merged)
