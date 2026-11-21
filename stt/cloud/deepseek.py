"""DeepSeek ASR tool API (2026-11-13): POST /v1/tools/asr, multipart `file` only.

No model / language / prompt parameters; WAV or Ogg Opus, <= 120 s, <= 8 MiB.
Docs: https://api-docs.deepseek.com/guides/asr
"""
import httpx

from .. import keys
from ..audio import Audio

URL = "https://api.deepseek.com/v1/tools/asr"


def ready() -> bool:
    return bool(keys.get("deepseek_api_key"))


def transcribe(audio: Audio) -> str:
    if audio.seconds > 120:
        raise ValueError("DeepSeek ASR accepts at most 120 s per request")
    r = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {keys.get('deepseek_api_key')}"},
        files={"file": ("audio.wav", audio.wav, "audio/wav")},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {' '.join(r.text.split())[:300]}")
    return r.json()["text"]
