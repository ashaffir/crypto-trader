import os
import tempfile
import asyncio

from src.logger import ParquetLogbook
from ui.lib.settings_state import load_llm_settings, save_llm_settings
from src.utils.llm_client import LLMClient, LLMConfig


async def _fake_server_response(client: LLMClient, payload: dict):
    # We won't actually hit network; instead we simulate generate() parsing paths
    # by calling generate() on a synthetic OpenAI-like JSON shape.
    # Monkeypatch client._client.post to return an object with .json() etc.
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            # Return a JSON array per new strict contract
            return {
                "choices": [
                    {
                        "message": {
                            "content": '[{"asset": "ETHUSDT", "direction": "buy", "leverage": 3, "confidence": 0.8}]'
                        }
                    }
                ]
            }

    class _Client:
        async def post(self, *a, **k):
            return _Resp()

        async def aclose(self):
            return None

    client._client = _Client()  # type: ignore
    return await client.generate(payload)


def test_llm_parsing_and_trade_log_write():
    cfg = LLMConfig(base_url="http://dummy")
    client = LLMClient(cfg)
    recs = asyncio.run(
        _fake_server_response(client, {"symbol": "ETHUSDT", "summary": {}})
    )
    assert recs is not None and isinstance(recs, list) and recs
    rec = recs[0]
    assert rec["asset"] == "ETHUSDT"
    assert rec["direction"] in ("buy", "sell", "long", "short")
    assert isinstance(rec["leverage"], int)

    with tempfile.TemporaryDirectory() as tmp:
        lb = ParquetLogbook(tmp)
        lb.append_trade_recommendation(
            [
                {
                    "ts_ms": 1,
                    "symbol": rec["asset"],
                    "asset": rec["asset"],
                    "direction": rec["direction"],
                    "leverage": int(rec["leverage"]),
                    "source": "llm",
                }
            ]
        )
        # Verify file created under expected path
        found = False
        for root, _dirs, files in os.walk(tmp):
            for f in files:
                if f.endswith(".parquet"):
                    found = True
                    break
        assert found


def test_ollama_generate_parsing(monkeypatch):
    # Simulate Ollama /api/generate server returning {response: "[ ... ]"}
    cfg = LLMConfig(
        base_url="https://llm.actappon.com", provider="Ollama", model="qwen2.5:7b"
    )
    client = LLMClient(cfg)

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": '[{"asset": "BTCUSDT", "direction": "buy", "leverage": 2}]'
            }

    class _Client:
        async def post(self, url, headers=None, json=None):  # type: ignore
            # Ensure endpoint normalization appends /api/generate
            assert url.endswith("/api/generate")
            return _Resp()

        async def aclose(self):
            return None

    client._client = _Client()  # type: ignore

    recs = asyncio.run(client.generate({"summary": {}}))
    assert isinstance(recs, list) and len(recs) == 1
    assert recs[0]["asset"] == "BTCUSDT"
    assert recs[0]["leverage"] == 2


def test_save_and_load_multiple_llm_configs(tmp_path, monkeypatch):
    # Redirect CONTROL_DIR used by settings_state to tmp
    monkeypatch.setenv("CONTROL_DIR", str(tmp_path))
    # Initial save with two configs, set active to second
    cfgs = {
        "default": {
            "base_url": "http://one",
            "api_key": "k1",
            "model": "m1",
            "system_prompt": "s1",
            "user_template": "u1",
        },
        "alt": {
            "base_url": "http://two",
            "api_key": "k2",
            "model": "m2",
            "system_prompt": "s2",
            "user_template": "u2",
        },
    }
    ok = save_llm_settings(
        {
            "active": "alt",
            "window_seconds": 45,
            "refresh_seconds": 7,
            "configs": cfgs,
        },
        base_dir=str(tmp_path),
    )
    assert ok
    loaded = load_llm_settings(base_dir=str(tmp_path))
    assert loaded.get("active") == "alt"
    assert int(loaded.get("window_seconds")) == 45
    assert int(loaded.get("refresh_seconds")) == 7
    lcfgs = loaded.get("configs")
    assert isinstance(lcfgs, dict)
    assert set(lcfgs.keys()) == {"default", "alt"}
    assert lcfgs["alt"]["base_url"] == "http://two"


def test_multi_asset_data_window_injection(monkeypatch):
    # Ensure the user template with {{DATA_WINDOW}} receives an array of summaries
    from src.utils.llm_client import LLMClient, LLMConfig

    captured_payload = {}

    # Configure client with OpenAI-compatible endpoint
    cfg = LLMConfig(base_url="https://api.deepseek.com", model="deepseek-chat")
    client = LLMClient(cfg)

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            # Return empty array to keep it simple
            return {"choices": [{"message": {"content": "[]"}}]}

    class _Client:
        async def post(self, url, headers=None, json=None):  # type: ignore
            nonlocal captured_payload
            captured_payload = json or {}
            return _Resp()

        async def aclose(self):
            return None

    client._client = _Client()  # type: ignore

    summaries = [
        {"symbol": "BTCUSDT", "count": 2, "series": {"mid": [1, 2]}},
        {"symbol": "ETHUSDT", "count": 3, "series": {"mid": [4, 5, 6]}},
    ]

    # Use variables exactly as app passes them
    recs = __import__("asyncio").run(
        client.generate(
            {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "window_seconds": 30,
                "DATA_WINDOW": summaries,
            }
        )
    )

    # Should parse to empty list
    assert isinstance(recs, list) and len(recs) == 0

    # Verify DATA_WINDOW was injected into the messages content
    msgs = (captured_payload or {}).get("messages", [])
    assert msgs and msgs[-1].get("role") == "user"
    content = msgs[-1].get("content") or ""
    # Both symbols should appear in the final rendered content
    assert "BTCUSDT" in content and "ETHUSDT" in content
