import os
import types
import asyncio
import pytest


@pytest.mark.asyncio
async def test_runtime_balances_http_error(monkeypatch, tmp_path):
    # Point CONTROL_DIR to temp
    control_dir = tmp_path / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CONTROL_DIR", str(control_dir))

    # Write runtime_config with fake keys to trigger our stub
    runtime_cfg_path = control_dir / "runtime_config.json"
    runtime_cfg_path.write_text(
        (
            '{"execution": {"venue": "spot", "network": "mainnet", '
            '"api_key": "k", "api_secret": "s"}}'
        ),
        encoding="utf-8",
    )

    # Stub the HTTP call path to raise HTTPStatusError
    import httpx
    from src.utils import binance_account as ba

    class _Resp:
        def json(self):
            return {"msg": "Invalid API-key, IP, or permissions"}

    def _raise(*args, **kwargs):
        req = httpx.Request("GET", "https://api.binance.com")
        resp = httpx.Response(401, request=req)
        # Attach our json content
        resp._content = b'{"msg":"Invalid API-key, IP, or permissions"}'
        raise httpx.HTTPStatusError("401", request=req, response=resp)

    async def _get_stub(client, url, headers):
        return _raise()

    monkeypatch.setattr(ba, "_get", _get_stub)

    out = await ba.fetch_balances_from_runtime_config()
    assert out.get("ok") is False
    assert out.get("error") == "http_error"
    d = out.get("detail")
    assert isinstance(d, dict)
    assert "msg" in d
