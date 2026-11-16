"""OpenAI /v1/audio/transcriptions (multipart)."""
import httpx

from . import keys
from .audio import Audio

URL = "https://api.openai.com/v1/audio/transcriptions"

# Steer toward Simplified Chinese characters even when the speech is Wu dialect.
PROMPT = "微信语音，说话人讲普通话或余姚话（吴语）。请用简体中文转写，加标点。"


def ready() -> bool:
    return bool(keys.get("openai_api_key"))


def transcribe(audio: Audio, model: str, prompt: str | None = PROMPT) -> str:
    data: dict = {"model": model}
    if prompt:
        data["prompt"] = prompt
    if model == "gpt-transcribe":
        data["languages[]"] = ["zh"]
    else:
        data["language"] = "zh"
    r = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {keys.get('openai_api_key')}"},
        data=data,
        files={"file": ("audio.wav", audio.wav, "audio/wav")},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {' '.join(r.text.split())[:300]}")
    return r.json()["text"]
