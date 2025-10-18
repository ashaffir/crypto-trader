from src.config import load_app_config


def test_load_app_config_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "config.yaml"))
    # empty file
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

    cfg = load_app_config()
    assert "logbook_dir" in cfg.storage
    assert cfg.symbols == ["BTCUSDT"]
    assert cfg.streams.aggTrade is True
