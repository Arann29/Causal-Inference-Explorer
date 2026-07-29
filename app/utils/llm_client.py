"""OpenRouter-backed LLM client abstractions."""

import os
import time
from abc import ABC, abstractmethod
from typing import Optional

import streamlit as st
from openai import OpenAI

from app.utils.llm_config import DEFAULT_MODEL_ID, DEFAULT_LLM_PROVIDER, get_model_spec_by_id

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
OPENROUTER_DEFAULT_TITLE = os.getenv("OPENROUTER_X_TITLE", "Causal Inference Explorer")


class LLMClient(ABC):
    """Strategy interface for LLM providers."""

    @abstractmethod
    def ask(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL_ID,
        temperature: float | None = 0.0,
        max_tokens: int | None = 2500,
    ) -> Optional[str]:
        """Send a prompt to the configured model and return the text response."""


class OpenRouterClient(LLMClient):
    """OpenRouter-compatible client using the OpenAI SDK."""

    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            st.error("OPENROUTER_API_KEY environment variable not found.")
            st.stop()

        default_headers = {
            "HTTP-Referer": OPENROUTER_DEFAULT_REFERER,
            "X-Title": OPENROUTER_DEFAULT_TITLE,
        }

        self.client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers=default_headers,
        )

    def ask(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL_ID,
        temperature: float | None = 0.0,
        max_tokens: int | None = 1500,
    ) -> Optional[str]:
        """Send a prompt through OpenRouter and return the first assistant message."""
        try:
            model_spec = get_model_spec_by_id(model)

            request_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }

            extra_body: dict[str, object] = {
                "provider": {
                    "require_parameters": True,
                    "allow_fallbacks": False,
                }
            }

            if max_tokens is not None and "max_tokens" in model_spec.supported_parameters:
                request_kwargs["max_tokens"] = max_tokens

            if temperature is not None and "temperature" in model_spec.supported_parameters:
                request_kwargs["temperature"] = temperature

            if model.endswith(":free") and "reasoning" in model_spec.supported_parameters:
                extra_body["reasoning"] = {"enabled": True}
            elif model.startswith("openai/gpt-") and "reasoning" in model_spec.supported_parameters:
                # OpenAI GPT reasoning models count hidden reasoning against the
                # output cap. For these experiments Claude is run without
                # extended thinking, so disable GPT hidden reasoning too; otherwise
                # GPT can spend the whole budget and return no visible answer.
                extra_body["reasoning"] = {"effort": "none", "exclude": True}

            response = None
            retry_delays = (10, 30, 60)
            for attempt in range(len(retry_delays) + 1):
                try:
                    response = self.client.chat.completions.create(**request_kwargs, extra_body=extra_body)
                    break
                except Exception as e:
                    if attempt >= len(retry_delays) or not self._is_retryable_error(e):
                        raise

                    delay = retry_delays[attempt]
                    st.warning(
                        f"OpenRouter provider temporarily unavailable for model={model}; "
                        f"retrying in {delay}s ({attempt + 1}/{len(retry_delays)})."
                    )
                    time.sleep(delay)

            choices = getattr(response, "choices", None)
            if not choices:
                raw_response = None
                try:
                    raw_response = response.model_dump()
                except Exception:
                    raw_response = str(response)
                st.error(f"OpenRouter returned no choices for model {model}. Raw response: {raw_response}")
                return None

            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            content = self._extract_message_text(message) if message is not None else None

            if content is None:
                raw_response = None
                try:
                    raw_response = response.model_dump()
                except Exception:
                    raw_response = str(response)

                st.error(
                    f"OpenRouter returned no final answer text for model {model}. "
                    f"Raw response: {raw_response}"
                )
                return None

            return content
        except Exception as e:
            st.error(f"OpenRouter API error for model={model}: {e}")
            return None

    def _is_retryable_error(self, error: Exception) -> bool:
        """Return True for transient provider or gateway failures."""
        status_code = getattr(error, "status_code", None)
        if status_code in {408, 409, 429, 500, 502, 503, 504}:
            return True

        error_text = str(error).lower()
        retryable_markers = (
            "overloaded_error",
            "temporarily unavailable",
            "provider returned error",
            "rate limit",
            "timeout",
            "try again",
            "503",
        )
        return any(marker in error_text for marker in retryable_markers)

    def _extract_message_text(self, message: object) -> Optional[str]:
        """Extract only the final visible text from a chat completion message."""
        candidate_fields = (
            "content",
            "text",
            "output_text",
        )

        for field_name in candidate_fields:
            value = getattr(message, field_name, None)
            text = self._normalize_text(value)
            if text:
                return text

        return None

    def _normalize_text(self, value: object) -> Optional[str]:
        """Convert nested SDK values into plain text when possible."""
        if value is None:
            return None

        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None

        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                part = self._normalize_text(item)
                if part:
                    parts.append(part)
            return "\n".join(parts).strip() or None

        if isinstance(value, dict):
            for key in ("content", "text"):
                if key in value:
                    text = self._normalize_text(value[key])
                    if text:
                        return text
            return None

        text = str(value).strip()
        return text or None


def get_llm_client(client_type: str = DEFAULT_LLM_PROVIDER) -> LLMClient:
    """Factory for the active LLM provider."""
    if client_type in {"openrouter", DEFAULT_LLM_PROVIDER}:
        return OpenRouterClient()

    raise ValueError(f"Unsupported LLM client type: {client_type}")
