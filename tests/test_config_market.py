from src.config import load_app_config
from src.utils.config_utils import resolve_config_path


def test_app_config_market_default(monkeypatch):
    # Ensure default path points to repo config
    monkeypatch.setenv("CONFIG_PATH", resolve_config_path())
    cfg = load_app_config()
    assert getattr(cfg, "market", "spot") in ("spot", "futures")
