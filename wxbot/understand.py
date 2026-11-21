"""Turn one source event into what the LLM sees: a line of text plus an optional image."""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from stt import transcribe

from .log import log
from .paths import DATA

# every transcribed voice: all candidates and the final Mandarin (the labeling page shows them as drafts)
VOICE_STT = os.path.join(DATA, "voice_stt.jsonl")
_VOICE_STT_LOCK = threading.Lock()
# voices of 关闭 chats: what the local models heard, kept on this machine for the labeling page's drafts only
# (never shown to the LLM, never sent anywhere); the bot itself uses it just to catch a spoken 开启
VOICE_LOCAL = os.path.join(DATA, "voice_local.jsonl")
LOCAL_MODELS = ("whisper-ft", "wu-ft")
_VOICE_LOCAL_LOCK = threading.Lock()


def transcribe_local(ev, models=LOCAL_MODELS) -> dict:
    """Run the ready local models on a voice of a 关闭 chat and record them in voice_local.jsonl.
    Returns {model: text} for the ones that heard something."""
    from stt import PROVIDERS
    names = [n for n in models if n in PROVIDERS and PROVIDERS[n].ready()]
    if not names or not ev.get("voice_path"):
        return {}
    with ThreadPoolExecutor(len(names)) as pool:
        res = dict(zip(names, pool.map(lambda n: transcribe(ev["voice_path"], n), names)))
    for n, r in res.items():
        if r.error:
            log(f"[{ev['chat']}] local {n} {ev['local_id']} failed: {r.error}")
    rec = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "chat": ev["chat"], "local_id": ev["local_id"],
           "key": os.path.basename(ev["voice_path"]), "candidates": {n: r.text or f"✗ {r.error}" for n, r in res.items()},
           "latency": max(r.latency for r in res.values())}
    with _VOICE_LOCAL_LOCK:
        with open(VOICE_LOCAL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {n: r.text for n, r in res.items() if r.text}

VIEWABLE = {"jpg", "jpeg", "png", "gif", "webp"}
PLACEHOLDER = {"emoji": "[表情]", "video": "[视频]", "location": "[位置]", "card": "[名片]", "call": "[通话]"}


def understand(ev, stt_provider="combo"):
    """Returns (text, image_path) or None for events the bot should ignore. stt_provider=None: no STT."""
    kind = ev["type"]
    if kind == "system":
        return None
    if kind == "text":
        return ev.get("text") or "", None
    if kind == "voice":
        if not ev.get("voice_path"):
            return "[语音，未能提取]", None
        if not stt_provider:
            return "[语音]", None
        r = transcribe(ev["voice_path"], stt_provider)
        log(f"stt {ev['local_id']}: {r.text or r.error} ({r.latency}s)")
        rec = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "chat": ev["chat"], "local_id": ev["local_id"],
               "key": os.path.basename(ev["voice_path"]), "provider": stt_provider, "text": r.text,
               "error": r.error, "latency": r.latency, **{k: v for k, v in r.extra.items()}}
        with _VOICE_STT_LOCK:
            with open(VOICE_STT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return (f"[语音] {r.text}" if r.text else "[语音，无法识别]"), None
    if kind == "image":
        path, fmt = ev.get("image_path"), ev.get("image_format")
        if path and fmt in VIEWABLE:
            return "[图片]", path
        return "[图片，无法查看]", None
    if kind == "app":
        if ev.get("quote"):
            q = ev["quote"]
            return f"{ev.get('text') or ''}\n（引用 {q.get('sender') or ''}：{q.get('text') or ''}）", None
        parts = [p for p in (ev.get("title"), ev.get("description"), ev.get("url")) if p]
        return "[链接/文件] " + " — ".join(parts) if parts else "[应用消息]", None
    return PLACEHOLDER.get(kind, f"[{kind}]"), None
