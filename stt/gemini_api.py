"""Google Gemini — audio understanding via generateContent with inline WAV.

Gemini is a general multimodal LLM, not a dedicated ASR, so it can do in one step what `combo` does in
two: hear 余姚话 and write the Mandarin meaning. Two prompts:
  MANDARIN  output the speaker's meaning as Mandarin (comparable to combo / the reference glosses)
  VERBATIM  plain transcript in Chinese characters
Docs: https://ai.google.dev/gemini-api/docs/audio
"""
import base64
import re

import httpx

from . import keys
from .audio import Audio

BASE = "https://generativelanguage.googleapis.com/v1beta"

MANDARIN = ("这是一条微信语音。说话人讲普通话，或者讲余姚话（浙江宁波余姚的吴语方言，和上海话、宁波话很接近）。"
            "请仔细听，把说话人实际说的内容翻译成自然的普通话简体中文，带标点。"
            "如果本来就是普通话，照原样转写。只输出这段普通话，不要解释。")
VERBATIM = "请把这段语音逐字转写成简体中文，带标点。只输出转写结果。"


def ready() -> bool:
    return bool(keys.get("gemini_api_key"))


def _headers() -> dict:
    return {"x-goog-api-key": keys.get("gemini_api_key"), "Content-Type": "application/json"}


def models() -> list[str]:
    """Model ids that accept generateContent (to pick the newest ones)."""
    r = httpx.get(f"{BASE}/models", headers=_headers(), params={"pageSize": 1000}, timeout=30)
    r.raise_for_status()
    return [m["name"].removeprefix("models/") for m in r.json()["models"]
            if "generateContent" in m.get("supportedGenerationMethods", [])]


def transcribe(audio: Audio, model: str, prompt: str = MANDARIN) -> str:
    body = {"contents": [{"role": "user", "parts": [
        {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(audio.wav).decode()}},
        {"text": prompt}]}]}
    r = httpx.post(f"{BASE}/models/{model}:generateContent", headers=_headers(), json=body, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {' '.join(r.text.split())[:300]}")
    cands = r.json().get("candidates") or []
    if not cands:
        raise RuntimeError(f"no candidates: {r.text[:200]}")
    parts = cands[0].get("content", {}).get("parts", [])
    # gemini-*-transcribe ignores the prompt and answers {"audioTranscription": {"text": "今 天 是 …"}}
    text = "".join(p.get("text", "") or p.get("audioTranscription", {}).get("text", "")
                   for p in parts if not p.get("thought"))
    return re.sub(r"(?<=[^\x00-\x7f]) (?=[^\x00-\x7f])", "", text).strip()  # drop spaces between CJK chars
