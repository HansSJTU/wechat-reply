"""Alibaba Cloud Model Studio (百炼 / DashScope) ASR, synchronous, base64 audio.

All three list 吴语 among supported Chinese dialects:
  qwen3-asr-flash            OpenAI-compatible chat completions, input_audio data URI
  fun-asr-flash-2026-06-15   DashScope multimodal-generation
  qwen-audio-3.1-asr-flash   same endpoint; converts dialect to Mandarin text by default
Host: {workspace}.cn-beijing.maas.aliyuncs.com when `aliyun_workspace_id` is set, else the legacy
dashscope.aliyuncs.com. Docs: https://help.aliyun.com/zh/model-studio/asr-model
"""
import base64

import httpx

from . import keys
from .audio import RATE, Audio


def ready() -> bool:
    return bool(keys.get("dashscope_api_key"))


def _base() -> str:
    ws = keys.get("aliyun_workspace_id")
    return f"https://{ws}.cn-beijing.maas.aliyuncs.com" if ws else "https://dashscope.aliyuncs.com"


def _headers() -> dict:
    return {"Authorization": f"Bearer {keys.get('dashscope_api_key')}", "Content-Type": "application/json"}


def _data_uri(audio: Audio) -> str:
    return "data:audio/wav;base64," + base64.b64encode(audio.wav).decode()


def _post(path: str, body: dict, extra_headers: dict | None = None) -> dict:
    r = httpx.post(_base() + path, headers={**_headers(), **(extra_headers or {})}, json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {' '.join(r.text.split())[:300]}")
    return r.json()


def qwen3(audio: Audio) -> str:
    body = {
        "model": "qwen3-asr-flash",
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": _data_uri(audio)}}]}],
        "stream": False,
        "asr_options": {"enable_itn": True},
    }
    return _post("/compatible-mode/v1/chat/completions", body)["choices"][0]["message"]["content"]


def multimodal(audio: Audio, model: str) -> str:
    """fun-asr-flash-* and qwen-audio-3.1-asr-flash share this request shape."""
    body = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": _data_uri(audio)}}]}]},
        "parameters": {"format": "wav", "sample_rate": str(RATE)},
    }
    out = _post("/api/v1/services/aigc/multimodal-generation/generation", body,
                {"X-DashScope-SSE": "disable"})["output"]
    if "text" in out:
        return out["text"]
    # chat-style output shape, just in case
    content = out["choices"][0]["message"]["content"]
    return content if isinstance(content, str) else "".join(c.get("text", "") for c in content)
