"""Provider registry and the single `transcribe()` entry point."""
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from . import (aliyun_api, deepseek_api, gemini_api, local_whisper_wu, local_wu, openai_api, refine, siliconflow_api, tencent_api,
               volc_api, xfyun_api)
from .audio import Audio


@dataclass
class Provider:
    name: str
    label: str
    run: Callable[[Audio], "str | tuple[str, dict]"]  # text, or (text, extra details)
    ready: Callable[[], bool]


def _p(name, label, run, ready):
    return name, Provider(name, label, run, ready)


# newest available 2026-11-14 (gemini_api.models()); gemini-omni-* rejects audio input on this key
GEMINI_FLASH, GEMINI_PRO, GEMINI_4 = "gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-4-carbon"

PROVIDERS: dict[str, Provider] = dict([
    # OpenAI — gpt-transcribe is current; the 4o/whisper models shut down 2027-02-26.
    _p("openai", "OpenAI gpt-transcribe", lambda a: openai_api.transcribe(a, "gpt-transcribe"), openai_api.ready),
    _p("openai-4o", "OpenAI gpt-4o-transcribe", lambda a: openai_api.transcribe(a, "gpt-4o-transcribe"), openai_api.ready),
    _p("openai-4o-mini", "OpenAI gpt-4o-mini-transcribe", lambda a: openai_api.transcribe(a, "gpt-4o-mini-transcribe"), openai_api.ready),
    _p("openai-whisper", "OpenAI whisper-1", lambda a: openai_api.transcribe(a, "whisper-1"), openai_api.ready),
    # DeepSeek ASR tool API
    _p("deepseek", "DeepSeek ASR", deepseek_api.transcribe, deepseek_api.ready),
    # Volcengine: 流式 (nostream websocket) / 录音文件 (submit+query) / 极速版, 1.0 vs 2.0
    _p("volc-stream-2", "火山 流式语音识别 2.0 (seedasr, nostream)",
       lambda a: volc_api.stream(a, "volc.seedasr.sauc.duration"), volc_api.ready),
    _p("volc-stream-1", "火山 流式语音识别大模型 1.0 (bigasr, nostream)",
       lambda a: volc_api.stream(a, "volc.bigasr.sauc.duration"), volc_api.ready),
    _p("volc-file-2", "火山 录音文件识别 2.0 (seedasr)",
       lambda a: volc_api.file(a, "volc.seedasr.auc"), volc_api.ready),
    _p("volc-file-1", "火山 录音文件识别大模型 1.0 (bigasr)",
       lambda a: volc_api.file(a, "volc.bigasr.auc"), volc_api.ready),
    _p("volc-flash", "火山 录音文件识别 极速版 (auc_turbo)",
       lambda a: volc_api.flash(a, "volc.bigasr.auc_turbo"), volc_api.ready),
    # Alibaba 百炼: all list 吴语
    _p("ali-qwen3", "阿里 qwen3-asr-flash", aliyun_api.qwen3, aliyun_api.ready),
    _p("ali-fun", "阿里 fun-asr-flash-2026-06-15",
       lambda a: aliyun_api.multimodal(a, "fun-asr-flash-2026-06-15"), aliyun_api.ready),
    _p("ali-audio31", "阿里 qwen-audio-3.1-asr-flash (方言→普通话)",
       lambda a: aliyun_api.multimodal(a, "qwen-audio-3.1-asr-flash"), aliyun_api.ready),
    # Tencent Cloud
    _p("tencent-dialect", "腾讯 一句话识别 16k_zh_dialect (23方言, 含上海话)",
       tencent_api.sentence, tencent_api.ready),
    _p("tencent-flash", "腾讯 极速版 16k_zh_en (大模型1.0)", tencent_api.flash, tencent_api.ready_flash),
    _p("tencent-2.0", "腾讯 录音文件 16k_zh_en_2.0 (大模型2.0, 含宁波话/吴语)", tencent_api.rec, tencent_api.ready),
    # iFlytek
    _p("xf-v2", "讯飞 语音听写大模型 V2.0 (202方言, 含宁波话)", xfyun_api.transcribe, xfyun_api.ready),
    _p("xf-dialect", "讯飞 方言大模型 slm/mulacc (普通话+202方言免切换)",
       lambda a: xfyun_api.transcribe(a, domain="slm", accent="mulacc"), xfyun_api.ready),
    # SiliconFlow 硅基流动
    _p("sf-sensevoice", "硅基流动 FunAudioLLM/SenseVoiceSmall",
       lambda a: siliconflow_api.transcribe(a, "FunAudioLLM/SenseVoiceSmall"), siliconflow_api.ready),
    _p("sf-telespeech", "硅基流动 TeleAI/TeleSpeechASR (电信, 方言)",
       lambda a: siliconflow_api.transcribe(a, "TeleAI/TeleSpeechASR"), siliconflow_api.ready),
    # Local, offline: WenetSpeech-Wu U2++ Conformer (needs sherpa-onnx + models/wu-u2pp-int8)
    _p("wu-local", "本地 WenetSpeech-Wu U2++ int8 (吴语, CPU)", local_wu.transcribe, local_wu.ready),
    _p("whisper-wu", "本地 Whisper-Medium-Wu (WenetSpeech-Wu 微调, MPS)", local_whisper_wu.transcribe,
       local_whisper_wu.ready),
    # Gemini: one multimodal call that hears 余姚话 and writes Mandarin (see gemini_api.MANDARIN)
    _p("gemini-flash", f"Gemini {GEMINI_FLASH} → 普通话",
       lambda a: gemini_api.transcribe(a, GEMINI_FLASH), gemini_api.ready),
    _p("gemini-pro", f"Gemini {GEMINI_PRO} → 普通话",
       lambda a: gemini_api.transcribe(a, GEMINI_PRO), gemini_api.ready),
    _p("gemini-4", f"Gemini {GEMINI_4} → 普通话",
       lambda a: gemini_api.transcribe(a, GEMINI_4), gemini_api.ready),
    _p("gemini-transcribe", "Gemini gemini-3.5-transcribe (dedicated ASR, ignores the prompt)",
       lambda a: gemini_api.transcribe(a, "gemini-3.5-transcribe"), gemini_api.ready),
])

# Speech APIs whose transcripts the `combo` provider hands to the LLM (see stt/refine.py).
COMBO_ASR = ["volc-file-2", "deepseek", "openai"]  # file-2: same output as flash at 0.8 vs 4.5 元/h


def _combo(audio: Audio) -> tuple[str, dict]:
    names = [n for n in COMBO_ASR if PROVIDERS[n].ready()]
    with ThreadPoolExecutor(len(names)) as pool:
        asr = list(pool.map(lambda n: transcribe(audio, n), names))
    extra = {"candidates": {r.provider: r.text or f"✗ {r.error}" for r in asr},
             "asr_latency": max(r.latency for r in asr)}
    texts = [r.text for r in asr if r.text]
    if not texts:
        raise RuntimeError("every speech API failed: " + "; ".join(f"{r.provider}: {r.error}" for r in asr))
    t = time.monotonic()
    try:
        text = refine.refine(texts)
    except Exception as e:  # LLM down: fall back to the first transcript rather than nothing
        extra["refine_error"] = f"{type(e).__name__}: {e}"
        text = texts[0]
    extra["llm_latency"] = round(time.monotonic() - t, 2)
    return text, extra


PROVIDERS["combo"] = Provider(
    "combo", "volc-file-2 + deepseek + openai → deepseek-flash 余姚话→普通话", _combo,
    lambda: refine.ready() and any(PROVIDERS[n].ready() for n in COMBO_ASR))


@dataclass
class Result:
    provider: str
    file: str
    text: str = ""
    error: str = ""
    latency: float = 0.0  # seconds, network + recognition
    audio_seconds: float = 0.0
    extra: dict = field(default_factory=dict)


def available() -> list[str]:
    """Providers whose API keys are configured."""
    return [n for n, p in PROVIDERS.items() if p.ready()]


def transcribe(audio: "Audio | str", provider: str = "combo") -> Result:
    """Transcribe one clip with one provider. Never raises for API errors; see Result.error."""
    if isinstance(audio, str):
        audio = Audio(audio)
    p = PROVIDERS[provider]
    res = Result(provider, audio.name, audio_seconds=round(audio.seconds, 2))
    if not p.ready():
        res.error = "no API key configured"
        return res
    t = time.monotonic()
    try:
        out = p.run(audio)
        if isinstance(out, tuple):
            out, res.extra = out
        res.text = (out or "").strip()
    except Exception as e:  # report per-provider, keep the others running
        res.error = f"{type(e).__name__}: {e}"
    res.latency = round(time.monotonic() - t, 2)
    return res
