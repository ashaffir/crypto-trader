from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import httpx


@dataclass
class LLMConfig:
    base_url: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    user_template: Optional[str] = None


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._client = httpx.AsyncClient(timeout=10.0)
        # Debug snapshot of the last interaction to aid UI troubleshooting
        self._last_debug: Dict[str, Any] = {}

    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception:
            pass

    async def generate(
        self, variables: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        # Build prompt
        system = (self.cfg.system_prompt or "You are a trading assistant.").strip()
        user = (
            self.cfg.user_template
            or "Return a JSON array of trade recommendations with fields asset, direction, leverage, confidence."
        ).strip()
        # Prepare DATA_WINDOW JSON string if template includes it
        data_window: Any = (
            variables.get("DATA_WINDOW") if isinstance(variables, dict) else None
        )
        if data_window is None:
            # Fallback to use summary as the window if present
            data_window = (
                variables.get("summary") if isinstance(variables, dict) else None
            )
        data_window_json: str = json.dumps(
            data_window if data_window is not None else {},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            formatted_user = user.format(**variables)
        except Exception:
            formatted_user = user
        # Replace {{DATA_WINDOW}} (which becomes {DATA_WINDOW} after .format) with strict JSON
        if "{DATA_WINDOW}" in formatted_user:
            formatted_user = formatted_user.replace("{DATA_WINDOW}", data_window_json)

        payload: Dict[str, Any] = {
            "model": self.cfg.model or "generic",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": formatted_user},
            ],
        }
        # Best-effort endpoint normalization for OpenAI-compatible providers
        url = (self.cfg.base_url or "").strip()
        if url.endswith("/"):
            url = url[:-1]
        lowered = url.lower()
        if lowered and ("completions" not in lowered):
            # Append default path if user only provided host
            if "/v" not in lowered:
                url = f"{url}/v1/chat/completions"
            else:
                # If version present but no completions path, append it
                if not lowered.endswith("/chat/completions"):
                    url = f"{url}/chat/completions"

        self._last_debug = {
            "endpoint": url,
            "has_api_key": bool(self.cfg.api_key),
            "model": self.cfg.model,
            "payload_chars": len(json.dumps(payload)),
        }
        try:
            resp = await self._client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            try:
                self._last_debug["status_code"] = getattr(resp, "status_code", None)
            except Exception:
                pass
        except Exception as e:
            self._last_debug["error"] = str(e)
            return None

        # Try to find JSON in common response shapes. We require strict JSON payloads.
        # OpenAI-like: {choices: [{message: {content: "..."}}]}
        text: Optional[str] = None
        try:
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message")
                if isinstance(msg, dict):
                    text = msg.get("content")
        except Exception:
            text = None
        if text is None:
            # Fallback: assume direct JSON in data["text"] or the body itself
            if isinstance(data, dict) and isinstance(data.get("text"), str):
                text = data.get("text")
            elif isinstance(data, (list, dict)):
                try:
                    text = json.dumps(data)
                except Exception:
                    text = None

        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            # Strict JSON required – do not try to heuristically extract
            self._last_debug["raw_text"] = text[:2000]
            return None

        # Normalize into list of recommendations
        recs: List[Dict[str, Any]] = []

        def _validate_row(obj: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
            try:
                asset = str(obj.get("asset"))
                direction_raw = str(obj.get("direction")).lower()
                if direction_raw not in ("buy", "sell", "long", "short"):
                    return False, None
                leverage = int(obj.get("leverage"))
                confidence_val = obj.get("confidence")
                confidence: Optional[float] = None
                if confidence_val is not None:
                    confidence = float(confidence_val)
                    if confidence < 0.0 or confidence > 1.0:
                        confidence = None
                out: Dict[str, Any] = {
                    "asset": asset,
                    "direction": direction_raw,
                    "leverage": leverage,
                }
                if confidence is not None:
                    out["confidence"] = confidence
                return True, out
            except Exception:
                return False, None

        if isinstance(parsed, dict):
            ok, row = _validate_row(parsed)
            if ok and row is not None:
                recs.append(row)
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    ok, row = _validate_row(item)
                    if ok and row is not None:
                        recs.append(row)

        if not recs:
            return None
        return recs

    def last_debug(self) -> Dict[str, Any]:
        return dict(self._last_debug)


__all__ = ["LLMClient", "LLMConfig"]
