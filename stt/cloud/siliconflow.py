"""SiliconFlow (硅基流动) — OpenAI-style POST /v1/audio/transcriptions, multipart `file` + `model`.

Models: FunAudioLLM/SenseVoiceSmall (Alibaba SenseVoice), TeleAI/TeleSpeechASR (China Telecom,
advertised for Chinese dialects). Docs: https://docs.siliconflow.cn/cn/api-reference/audio/create-audio-transcriptions
"""
import httpx

from .. import keys
from ..audio import Audio

URL = "https://api.siliconflow.cn/v1/audio/transcriptions"


def ready() -> bool:
    return bool(keys.get("siliconflow_api_key"))


def transcribe(audio: Audio, model: str) -> str:
    r = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {keys.get('siliconflow_api_key')}"},
        files={"file": ("audio.wav", audio.wav, "audio/wav")},
        data={"model": model},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {' '.join(r.text.split())[:300]}")
    return r.json()["text"]
