"""Minimal LLM API client helpers for autonomous research orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMClientError(RuntimeError):
    """Raised when an LLM API call fails or returns invalid output."""


@dataclass(frozen=True)
class LLMGeneration:
    payload: dict[str, Any]
    raw_text: str
    model: str
    response_id: str | None
    usage: dict[str, Any]


@dataclass(frozen=True)
class LLMRawGeneration:
    raw_text: str
    model: str
    response_id: str | None
    usage: dict[str, Any]


def _strip_code_fences(text: str) -> str:
    raw = str(text).strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return raw


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = _strip_code_fences(text)
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise LLMClientError("LLM response did not contain a JSON object")
    snippet = raw[start : end + 1]
    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise LLMClientError(f"Invalid JSON in LLM response: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMClientError("LLM response JSON must be an object")
    return payload


def _extract_chat_content(choice_message: Any) -> str:
    if isinstance(choice_message, dict):
        content = choice_message.get("content")
    else:
        content = None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                txt = part.get("text")
                if txt is not None:
                    chunks.append(str(txt))
        return "\n".join(chunks).strip()
    return ""


class OpenAIChatJSONClient:
    """Small OpenAI-compatible chat-completions JSON client."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 90.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
    ) -> None:
        key = api_key or os.getenv("LLM_API_KEY")
        if not key:
            raise LLMClientError("LLM_API_KEY is required")
        self._api_key = str(key)
        self._model = str(model).strip()
        if not self._model:
            raise LLMClientError("model is required")
        self._base_url = str(base_url).rstrip("/")
        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff_seconds = max(0.1, float(retry_backoff_seconds))

    @property
    def model(self) -> str:
        return self._model

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{endpoint}"
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        attempt = 0
        while True:
            try:
                with urlopen(req, timeout=self._timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise LLMClientError("LLM API returned non-object response")
                return parsed
            except HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = str(exc)
                if attempt >= self._max_retries:
                    raise LLMClientError(f"LLM HTTP {exc.code}: {detail[:500]}") from exc
            except (URLError, TimeoutError, json.JSONDecodeError, LLMClientError) as exc:
                if attempt >= self._max_retries:
                    raise LLMClientError(f"LLM request failed: {exc}") from exc
            attempt += 1
            time.sleep(self._retry_backoff_seconds * attempt)

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 2500,
    ) -> LLMGeneration:
        raw = self.generate_raw(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            force_json_object=True,
        )
        json_payload = _extract_json_object(raw.raw_text)
        return LLMGeneration(
            payload=json_payload,
            raw_text=raw.raw_text,
            model=raw.model,
            response_id=raw.response_id,
            usage=raw.usage,
        )

    def generate_raw(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 2500,
        force_json_object: bool = True,
    ) -> LLMRawGeneration:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(user_prompt)},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_output_tokens),
        }
        if force_json_object:
            payload["response_format"] = {"type": "json_object"}
        data = self._post_json("/chat/completions", payload)

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMClientError("LLM response missing choices")
        choice0 = choices[0]
        message = choice0.get("message") if isinstance(choice0, dict) else None
        raw_text = _extract_chat_content(message)
        if not raw_text:
            raise LLMClientError("LLM response contained no message content")

        usage = data.get("usage")
        usage_obj = usage if isinstance(usage, dict) else {}
        response_id = data.get("id")
        return LLMRawGeneration(
            raw_text=raw_text,
            model=self._model,
            response_id=str(response_id) if response_id is not None else None,
            usage=dict(usage_obj),
        )


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from model text output."""
    return _extract_json_object(text)


__all__ = [
    "LLMClientError",
    "LLMGeneration",
    "LLMRawGeneration",
    "OpenAIChatJSONClient",
    "extract_json_object",
]
