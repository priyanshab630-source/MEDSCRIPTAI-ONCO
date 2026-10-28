"""Thin LLM client - Sarvam AI by default, any OpenAI-compatible endpoint.

If no API key is configured, `available()` is False and every agent falls back
to deterministic rule-based reasoning. That is why the prototype runs with no
keys, and why the evidence trail stays reproducible either way.
"""

import sys
import requests

import config


def available() -> bool:
    return bool(config.SARVAM_API_KEY)


def chat(system: str, prompt: str, temperature: float = 0.2):
    """Returns the model's text, or None if the LLM is unavailable/failed."""
    if not available():
        return None

    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": config.LLM_MAX_TOKENS,
    }
    headers = {
        "api-subscription-key": config.SARVAM_API_KEY,
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            config.LLM_BASE_URL, json=payload, headers=headers,
            timeout=config.LLM_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:                     # noqa: BLE001 - never break a run
        print(f"[llm] call failed ({exc}); falling back to rule-based reasoning", file=sys.stderr)
        return None
