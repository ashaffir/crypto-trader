from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import httpx
from loguru import logger


@dataclass
class LLMConfig:
    base_url: str
    provider: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    user_template: Optional[str] = None


class LLMClient:
    def __init__(self, cfg: LLMConfig, timeout: float = 60.0) -> None:
        self.cfg = cfg
        self._client = httpx.AsyncClient(timeout=timeout)
        # Debug snapshot of the last interaction to aid UI troubleshooting
        self._last_debug: Dict[str, Any] = {}
        logger.debug(f"LLMClient initialized with timeout={timeout}s")
        # Optional: path to write last request for debugging (set via setter)
        self._debug_save_path: Optional[str] = None
        # Derived response path when debug save is enabled
        self._debug_response_path: Optional[str] = None

    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception:
            pass

    async def generate(
        self, variables: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        logger.info(f"LLM generate called with model={self.cfg.model}")

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        provider = (self.cfg.provider or "").strip().lower()
        # Do not send Authorization for Ollama (server currently unauthenticated)
        if self.cfg.api_key and provider != "ollama":
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        # Build prompt
        system = (self.cfg.system_prompt or "You are a trading assistant.").strip()
        user = (
            self.cfg.user_template
            or "Return a JSON array of trade recommendations with fields asset, direction, leverage, confidence."
        ).strip()

        logger.debug(f"System prompt length: {len(system)} chars")
        logger.debug(f"User template length: {len(user)} chars")
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

        # provider is already computed above

        # Build payload and endpoint based on provider
        if provider == "ollama":
            # Ollama /api/generate expects a flat prompt string and returns {response: "..."}
            # Combine system and user content to form a single prompt
            prompt_parts: List[str] = []
            if system:
                prompt_parts.append(system)
            if formatted_user:
                prompt_parts.append(formatted_user)
            prompt_text = "\n\n".join(prompt_parts)

            payload = {
                "model": (self.cfg.model or "qwen2.5:7b"),
                "prompt": prompt_text,
                "stream": False,
            }

            url = (self.cfg.base_url or "").strip()
            if url.endswith("/"):
                url = url[:-1]
            lowered = url.lower()
            if lowered and ("/api/generate" not in lowered):
                url = f"{url}/api/generate"
        else:
            # Default: OpenAI-compatible chat completions
            payload = {
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
            "provider": provider or None,
            "has_api_key": bool(self.cfg.api_key),
            "model": self.cfg.model,
            "payload_chars": len(json.dumps(payload)),
        }

        # If debug path is set, write last request (overwrite)
        try:
            if self._debug_save_path:
                with open(self._debug_save_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"endpoint": url, "headers": headers, "payload": payload}, f
                    )
        except Exception:
            pass

        logger.info(f"Sending request to {url}")
        logger.debug(f"Payload size: {self._last_debug['payload_chars']} chars")

        try:
            resp = await self._client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            logger.info(
                f"Received response: status={getattr(resp, 'status_code', None)}"
            )

            try:
                self._last_debug["status_code"] = getattr(resp, "status_code", None)
                # Capture full response for debugging
                self._last_debug["response_body"] = data
            except Exception:
                pass
            # Best-effort: persist raw response to file if debug is enabled
            try:
                if self._debug_response_path:
                    with open(self._debug_response_path, "w", encoding="utf-8") as f:
                        json.dump(data, f)
            except Exception:
                pass
        except httpx.TimeoutException as e:
            error_msg = f"Request timeout: {str(e)}"
            logger.error(error_msg)
            self._last_debug["error"] = error_msg
            return None
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
            logger.error(f"LLM request failed: {error_msg}")
            self._last_debug["error"] = error_msg
            self._last_debug["status_code"] = e.response.status_code
            return None
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"LLM request failed: {error_msg}")
            logger.exception("Full traceback:")
            self._last_debug["error"] = error_msg
            self._last_debug["exception_type"] = type(e).__name__
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
        # Ollama /api/generate: {response: "..."} or /api/chat: {message: {content: "..."}}
        if text is None:
            try:
                if isinstance(data, dict) and isinstance(data.get("response"), str):
                    text = data.get("response")
                elif isinstance(data, dict) and isinstance(data.get("message"), dict):
                    msg = data.get("message")
                    text = (
                        msg.get("content")
                        if isinstance(msg.get("content"), str)
                        else None
                    )
            except Exception:
                pass
        if text is None:
            # Fallback: assume direct JSON in data["text"] or the body itself
            if isinstance(data, dict) and isinstance(data.get("text"), str):
                text = data.get("text")
            elif isinstance(data, (list, dict)):
                try:
                    text = json.dumps(data)
                except Exception:
                    text = None

        self._last_debug["extracted_text"] = text[:2000] if text else None

        if not text:
            logger.warning("No text content extracted from LLM response")
            self._last_debug["failure_reason"] = (
                "No text content extracted from response"
            )
            return None

        logger.debug(
            f"Extracted text: {text[:200]}..."
            if len(text) > 200
            else f"Extracted text: {text}"
        )

        try:
            parsed = json.loads(text)
            self._last_debug["parsed_json"] = parsed
            logger.info(
                f"Successfully parsed JSON: {type(parsed).__name__} with {len(parsed) if isinstance(parsed, list) else 1} item(s)"
            )
        except Exception as parse_err:
            # Strict JSON required – do not try to heuristically extract
            logger.error(f"JSON parse error: {parse_err}")
            self._last_debug["raw_text"] = text[:2000]
            self._last_debug["failure_reason"] = f"JSON parse error: {parse_err}"
            return None

        # Normalize into list of recommendations
        recs: List[Dict[str, Any]] = []
        validation_failures: List[str] = []

        def _validate_row(obj: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
            try:
                asset = str(obj.get("asset"))
                direction_raw = str(obj.get("direction")).lower()
                if direction_raw not in ("buy", "sell", "long", "short"):
                    validation_failures.append(f"Invalid direction: {direction_raw}")
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
            except Exception as val_err:
                validation_failures.append(f"Validation error: {val_err}")
                return False, None

        if isinstance(parsed, dict):
            ok, row = _validate_row(parsed)
            if ok and row is not None:
                recs.append(row)
        elif isinstance(parsed, list):
            # Empty list is valid (means no opportunities)
            if not parsed:
                logger.info("Parsed recommendations: []")
                return []
            for item in parsed:
                if isinstance(item, dict):
                    ok, row = _validate_row(item)
                    if ok and row is not None:
                        recs.append(row)
        else:
            # Parsed is neither dict nor list - this is invalid
            self._last_debug["failure_reason"] = (
                "Invalid response format (not dict or list)"
            )
            return None

        # If we parsed items but none were valid, that's a problem
        if isinstance(parsed, list) and parsed and not recs:
            logger.warning(f"Validation failed: {len(parsed)} items parsed, 0 valid")
            logger.warning(f"Validation errors: {validation_failures}")
            self._last_debug["failure_reason"] = (
                "No valid recommendations after validation"
            )
            self._last_debug["validation_failures"] = validation_failures
            return None

        # Printout of the parsed/validated trade data for visibility
        try:
            recs_json = json.dumps(recs, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            recs_json = str(recs)
        logger.info(f"Parsed recommendations: {recs_json}")

        logger.info(f"Returning {len(recs)} valid recommendations")
        return recs

    def last_debug(self) -> Dict[str, Any]:
        return dict(self._last_debug)

    def set_debug_save_path(self, path: Optional[str]) -> None:
        self._debug_save_path = path
        try:
            if path:
                base_dir = os.path.dirname(path)
                # Always co-locate response file next to request file
                self._debug_response_path = os.path.join(
                    base_dir, "llm_last_response.json"
                )
            else:
                self._debug_response_path = None
        except Exception:
            # Fall back to disabling response save if any issue computing path
            self._debug_response_path = None


__all__ = ["LLMClient", "LLMConfig"]
