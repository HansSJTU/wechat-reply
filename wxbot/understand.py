"""Turn one source event into what the LLM sees: a line of text plus an optional image."""
import json
import os
import time

from stt import transcribe

from .log import log
from .paths import DATA

# every transcribed voice: all candidates and the final Mandarin (the labeling page shows them as drafts)
VOICE_STT = os.path.join(DATA, "voice_stt.jsonl")

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
