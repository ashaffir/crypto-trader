import os
import json
import asyncio

from src.utils.llm_client import LLMClient, LLMConfig


def test_debug_files_written(tmp_path):
    # Configure client and enable debug save into tmp control dir
    req_path = os.path.join(str(tmp_path), "llm_last_request.json")

    cfg = LLMConfig(base_url="http://dummy")
    client = LLMClient(cfg)
    client.set_debug_save_path(req_path)

    # Fake HTTP client returning OpenAI-like response
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '[{"asset":"BTCUSDT","direction":"buy","leverage":1}]'
                        }
                    }
                ]
            }

    class _Client:
        async def post(self, *a, **k):  # type: ignore
            return _Resp()

        async def aclose(self):
            return None

    client._client = _Client()  # type: ignore

    # Run generate to trigger file writes
    recs = asyncio.run(client.generate({}))
    assert recs is not None

    # Verify request file written
    assert os.path.exists(req_path)
    with open(req_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
        assert isinstance(doc, dict) and "endpoint" in doc and "payload" in doc

    # Verify response file written next to it
    resp_path = os.path.join(str(tmp_path), "llm_last_response.json")
    assert os.path.exists(resp_path)
    with open(resp_path, "r", encoding="utf-8") as f:
        body = json.load(f)
        assert isinstance(body, dict)
