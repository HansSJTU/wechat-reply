"""Provider registry and the single `transcribe()` entry point."""
import importlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from . import local_whisper_wu, local_wu, refine
from .audio import Audio


@dataclass
class Provider:
    name: str
    label: str
    run: Callable[[Audio], "str | tuple[str, dict]"]  # text, or (text, extra details)
    ready: Callable[[], bool]


def _p(name, label, run, ready):
    return name, Provider(name, label, run, ready)


def _cloud(module: str, fn: str = "transcribe", *args, ready: str = "ready", **kw):
    """(run, ready) for a function of stt/cloud/<module>.py, imported on first use."""
    def mod():
        return importlib.import_module(f"{__package__}.cloud.{module}")
    return (lambda a: getattr(mod(), fn)(a, *args, **kw)), (lambda: getattr(mod(), ready)())


# newest available 2026-11-14 (cloud.gemini.models()); gemini-omni-* rejects audio input on this key
GEMINI_FLASH, GEMINI_PRO, GEMINI_4 = "gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-4-carbon"

PROVIDERS: dict[str, Provider] = dict([
    # Local, offline (audio never leaves the machine): WenetSpeech-Wu U2++ Conformer (sherpa-onnx) and
    # Whisper-Medium-Wu, each as released and fine-tuned on the labeled clips (finetune/).
    _p("wu-local", "本地 WenetSpeech-Wu U2++ int8 (吴语, CPU)", local_wu.transcribe, local_wu.ready),
    _p("wu-ft", "本地 U2++ 微调版 int8 (用标注的微信语音微调, CPU)",
       lambda a: local_wu.transcribe(a, local_wu.FT_DIR), lambda: local_wu.ready(local_wu.FT_DIR)),
    _p("whisper-wu", "本地 Whisper-Medium-Wu (WenetSpeech-Wu 微调, MPS)", local_whisper_wu.transcribe,
       local_whisper_wu.ready),
    _p("whisper-ft", "本地 Whisper-Medium-Wu 微调版 (用标注的语音微调, MPS)",
       lambda a: local_whisper_wu.transcribe(a, ckpt=local_whisper_wu.FT_CKPT),
       lambda: local_whisper_wu.ready(local_whisper_wu.FT_CKPT)),
    # ---- cloud (stt/cloud/), for comparison only
    # OpenAI — gpt-transcribe is current; the 4o/whisper models shut down 2027-02-26.
    _p("openai", "OpenAI gpt-transcribe", *_cloud("openai", "transcribe", "gpt-transcribe")),
    _p("openai-4o", "OpenAI gpt-4o-transcribe", *_cloud("openai", "transcribe", "gpt-4o-transcribe")),
    _p("openai-4o-mini", "OpenAI gpt-4o-mini-transcribe", *_cloud("openai", "transcribe", "gpt-4o-mini-transcribe")),
    _p("openai-whisper", "OpenAI whisper-1", *_cloud("openai", "transcribe", "whisper-1")),
    # DeepSeek ASR tool API
    _p("deepseek", "DeepSeek ASR", *_cloud("deepseek")),
    # Volcengine: 流式 (nostream websocket) / 录音文件 (submit+query) / 极速版, 1.0 vs 2.0
    _p("volc-stream-2", "火山 流式语音识别 2.0 (seedasr, nostream)", *_cloud("volc", "stream", "volc.seedasr.sauc.duration")),
    _p("volc-stream-1", "火山 流式语音识别大模型 1.0 (bigasr, nostream)", *_cloud("volc", "stream", "volc.bigasr.sauc.duration")),
    _p("volc-file-2", "火山 录音文件识别 2.0 (seedasr)", *_cloud("volc", "file", "volc.seedasr.auc")),
    _p("volc-file-1", "火山 录音文件识别大模型 1.0 (bigasr)", *_cloud("volc", "file", "volc.bigasr.auc")),
    _p("volc-flash", "火山 录音文件识别 极速版 (auc_turbo)", *_cloud("volc", "flash", "volc.bigasr.auc_turbo")),
    # Alibaba 百炼: all list 吴语
    _p("ali-qwen3", "阿里 qwen3-asr-flash", *_cloud("aliyun", "qwen3")),
    _p("ali-fun", "阿里 fun-asr-flash-2026-06-15", *_cloud("aliyun", "multimodal", "fun-asr-flash-2026-06-15")),
    _p("ali-audio31", "阿里 qwen-audio-3.1-asr-flash (方言→普通话)",
       *_cloud("aliyun", "multimodal", "qwen-audio-3.1-asr-flash")),
    # Tencent Cloud
    _p("tencent-dialect", "腾讯 一句话识别 16k_zh_dialect (23方言, 含上海话)", *_cloud("tencent", "sentence")),
    _p("tencent-flash", "腾讯 极速版 16k_zh_en (大模型1.0)", *_cloud("tencent", "flash", ready="ready_flash")),
    _p("tencent-2.0", "腾讯 录音文件 16k_zh_en_2.0 (大模型2.0, 含宁波话/吴语)", *_cloud("tencent", "rec")),
    # iFlytek
    _p("xf-v2", "讯飞 语音听写大模型 V2.0 (202方言, 含宁波话)", *_cloud("xfyun")),
    _p("xf-dialect", "讯飞 方言大模型 slm/mulacc (普通话+202方言免切换)",
       *_cloud("xfyun", domain="slm", accent="mulacc")),
    # SiliconFlow 硅基流动
    _p("sf-sensevoice", "硅基流动 FunAudioLLM/SenseVoiceSmall",
       *_cloud("siliconflow", "transcribe", "FunAudioLLM/SenseVoiceSmall")),
    _p("sf-telespeech", "硅基流动 TeleAI/TeleSpeechASR (电信, 方言)",
       *_cloud("siliconflow", "transcribe", "TeleAI/TeleSpeechASR")),
    # Gemini: one multimodal call that hears 余姚话 and writes Mandarin (see cloud.gemini.MANDARIN)
    _p("gemini-flash", f"Gemini {GEMINI_FLASH} → 普通话", *_cloud("gemini", "transcribe", GEMINI_FLASH)),
    _p("gemini-pro", f"Gemini {GEMINI_PRO} → 普通话", *_cloud("gemini", "transcribe", GEMINI_PRO)),
    _p("gemini-4", f"Gemini {GEMINI_4} → 普通话", *_cloud("gemini", "transcribe", GEMINI_4)),
    _p("gemini-transcribe", "Gemini gemini-3.5-transcribe (dedicated ASR, ignores the prompt)",
       *_cloud("gemini", "transcribe", "gemini-3.5-transcribe")),
])

# Speech APIs whose transcripts the `combo` provider hands to the LLM (see stt/refine.py).
# Since 2026-11-19 only the local fine-tuned models: the audio never leaves this machine, only their
# text goes to the LLM. On the labeled test clips whisper-ft + wu-ft as LLM input beat either alone; the
# cloud ASRs (volc-file-2, deepseek, openai) were dropped after they misheard family 余姚话 voices.
COMBO_ASR = ["whisper-ft", "wu-ft"]
# Of those, the ones the LLM sees (all are recorded as candidates). Since 2026-11-19 only whisper-ft: wu-ft's
# near-homophone characters (侬人则个母轻头啦 for 侬认识个母亲头啦) pulled the Mandarin off. The others are
# used only when these all fail.
COMBO_LLM = ["whisper-ft"]


def _combo(audio: Audio) -> tuple[str, dict]:
    names = [n for n in COMBO_ASR if PROVIDERS[n].ready()]
    with ThreadPoolExecutor(len(names)) as pool:
        asr = list(pool.map(lambda n: transcribe(audio, n), names))
    extra = {"candidates": {r.provider: r.text or f"✗ {r.error}" for r in asr},
             "asr_latency": max(r.latency for r in asr)}
    texts = [r.text for r in asr if r.text and r.provider in COMBO_LLM] or [r.text for r in asr if r.text]
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
    "combo", "local whisper-ft (+ wu-ft recorded) → deepseek-flash 余姚话→普通话", _combo,
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
