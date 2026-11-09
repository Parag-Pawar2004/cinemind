"""Provider-agnostic LLM client using only free options, via plain HTTP (no SDKs).

auto order: Groq -> Gemini -> OpenRouter(:free) -> local Ollama -> None (rule-based agents)
Any failure returns None so the agents fall back to deterministic logic.
"""
from __future__ import annotations

import json
import re

import requests

from . import config

_last_error: str | None = None


def available_providers() -> list[str]:
    out = []
    if config.get("GROQ_API_KEY"):
        out.append("groq")
    if config.get("GEMINI_API_KEY"):
        out.append("gemini")
    if config.get("OPENROUTER_API_KEY"):
        out.append("openrouter")
    try:
        requests.get(config.get("OLLAMA_URL") + "/api/tags", timeout=0.6)
        out.append("ollama")
    except requests.RequestException:
        pass
    return out


def active_provider() -> str | None:
    pref = (config.get("LLM_PROVIDER") or "auto").lower()
    if pref == "none":
        return None
    avail = available_providers()
    if pref != "auto":
        return pref if pref in avail else None
    return avail[0] if avail else None


def _openai_style(url, key, model, messages, json_mode, extra_headers=None):
    body = {"model": model, "messages": messages, "temperature": 0.3}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    hdr = {"Authorization": f"Bearer {key}", **(extra_headers or {})}
    r = requests.post(url, json=body, timeout=45, headers=hdr)
    if r.status_code == 400 and json_mode:  # model may not support JSON mode: retry plain
        body.pop("response_format", None)
        body["messages"] = messages + [{"role": "user", "content": "Reply with valid JSON only."}]
        r = requests.post(url, json=body, timeout=45, headers=hdr)
    if r.status_code >= 400:
        raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"]


def chat(messages: list[dict], json_mode: bool = False, provider: str | None = None) -> str | None:
    global _last_error
    p = provider or active_provider()
    if not p:
        return None
    try:
        if p == "groq":
            return _openai_style("https://api.groq.com/openai/v1/chat/completions",
                                 config.get("GROQ_API_KEY"), config.get("GROQ_MODEL"), messages, json_mode)
        if p == "openrouter":
            return _openai_style("https://openrouter.ai/api/v1/chat/completions",
                                 config.get("OPENROUTER_API_KEY"), config.get("OPENROUTER_MODEL"),
                                 messages, json_mode, {"X-Title": "CineMind"})
        if p == "gemini":
            sys_txt = "\n".join(m["content"] for m in messages if m["role"] == "system")
            contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                        for m in messages if m["role"] != "system"]
            body = {"contents": contents, "generationConfig": {"temperature": 0.3}}
            if sys_txt:
                body["systemInstruction"] = {"parts": [{"text": sys_txt}]}
            if json_mode:
                body["generationConfig"]["responseMimeType"] = "application/json"
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{config.get('GEMINI_MODEL')}:generateContent")
            r = requests.post(url, json=body, timeout=45, headers={"x-goog-api-key": config.get("GEMINI_API_KEY")})
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        if p == "ollama":
            body = {"model": config.get("OLLAMA_MODEL"), "messages": messages, "stream": False}
            if json_mode:
                body["format"] = "json"
            r = requests.post(config.get("OLLAMA_URL") + "/api/chat", json=body, timeout=120)
            r.raise_for_status()
            return r.json()["message"]["content"]
    except Exception as e:  # noqa: BLE001
        _last_error = f"{p}: {type(e).__name__}: {str(e)[:200]}"
        return None
    return None


def chat_json(messages: list[dict]) -> dict | list | None:
    txt = chat(messages, json_mode=True)
    return extract_json(txt) if txt else None


def extract_json(txt: str):
    try:
        return json.loads(txt)
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"(\{.*\}|\[.*\])", txt or "", re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return None
    return None


def last_error() -> str | None:
    return _last_error
