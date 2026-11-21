"""Data for the chat viewer (/chats on the labeling server): what the WeChat bot saw, said and heard.

Merges, per watched chat:
  data/messages.jsonl     every recorded message (the source logs all watched chats)
  data/bot.db             the bot's replies (native sends never reach WeChat's DB) and the 开启/关闭 state
  data/voice_stt.jsonl    the bot's transcription of each voice while the chat was 开启 (candidates + Mandarin)
  data/labels/            local model drafts and the human label of each voice
开启 / 关闭 markers are read back from the command messages, parsed as wxbot/pipeline.py does.
Stdlib only (config via wxbot/config.py); everything stays on this machine.
"""
import json
import os
import re
import sqlite3

from labeling.dataset import DRAFTS, LABELS, VOICE_STT
from stt.paths import DATA
from wxbot.config import load_config, switch_words

MENTION = re.compile(r"@[^\s ]+[\s ]*")


def _jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _db():
    p = os.path.join(DATA, "bot.db")
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True) if os.path.exists(p) else None


def _default_on(cfg, chat, name):
    """State before the first 开启/关闭, as wxbot/pipeline.py: `chats`/`mention` on, `listen` off."""
    for key, on in (("chats", True), ("mention", True), ("listen", False)):
        if chat in (cfg.get(key) or []) or name in (cfg.get(key) or []):
            return on
    return False


def _command(ev, on_words, off_words):
    if ev.get("type") != "text" or ev.get("is_self"):
        return None
    text = ev.get("text") or ""
    if ev.get("is_group"):
        if not ev.get("mentions_me"):
            return None
        text = MENTION.sub("", text)
    word = text.strip(" 。.!！~～")
    return "on" if word in on_words else "off" if word in off_words else None


def _switch_row(con, chat):
    try:
        row = con.execute("select on_, at from switches where chat = ?", (chat,)).fetchone()
    except sqlite3.OperationalError:  # older bot.db without the table
        return None
    return row


def chats() -> list[dict]:
    cfg = load_config()
    msgs = _jsonl(os.path.join(DATA, "messages.jsonl"))
    con = _db()
    by_chat = {}
    for m in msgs:
        c = by_chat.setdefault(m["chat"], {"chat": m["chat"], "name": m.get("chat_name") or m["chat"],
                                           "is_group": m.get("is_group", False), "last": None, "count": 0})
        c["count"] += 1
        c["last"] = m
    out = []
    for c in by_chat.values():
        row = con and _switch_row(con, c["chat"])
        on = bool(row[0]) if row else _default_on(cfg, c["chat"], c["name"])
        last = c.pop("last")
        out.append({**c, "on": on, "switched_at": row[1] if row else None, "last_time": last["time"],
                    "last_preview": _preview(last)})
    return sorted(out, key=lambda c: -c["last_time"])


def _preview(m):
    t = m.get("type")
    if t == "text":
        return m.get("text") or ""
    return {"voice": "[语音]", "image": "[图片]", "video": "[视频]", "emoji": "[表情]"}.get(t, f"[{t}]")


def chat(chat_id: str) -> dict:
    cfg = load_config()
    on_words, off_words = switch_words(cfg)
    msgs = [m for m in _jsonl(os.path.join(DATA, "messages.jsonl")) if m["chat"] == chat_id]
    stt = {r["key"]: r for r in _jsonl(VOICE_STT) if r.get("chat") == chat_id}
    drafts = json.load(open(DRAFTS, encoding="utf-8")) if os.path.exists(DRAFTS) else {}
    labels = {}
    for r in _jsonl(LABELS):
        labels[r["key"]] = r
    # what the reply LLM saw for each message (bot.db, by local_id): the transcription of voices from before
    # voice_stt.jsonl existed
    seen = {}
    con = _db()
    if con:
        for text, meta in con.execute("select text, meta from messages where chat = ? and role = 'user'", (chat_id,)):
            m = json.loads(meta or "{}")
            if m.get("local_id") is not None:
                seen[m["local_id"]] = text
    name = msgs[0].get("chat_name") if msgs else chat_id
    is_group = chat_id.endswith("@chatroom")
    items, switches = [], []
    for m in msgs:
        it = {"kind": "msg", "time": m["time"], "type": m["type"], "sender": m.get("sender_name") or m.get("sender"),
              "self": m.get("is_self", False), "text": m.get("text"), "mentions_me": m.get("mentions_me", False)}
        if m["type"] == "voice" and m.get("voice_path"):
            key = os.path.basename(m["voice_path"])
            r, lab = stt.get(key), labels.get(key)
            old = re.sub(r"^.*?\[语音\]\s*", "", seen.get(m["local_id"], "")) if "[语音] " in seen.get(m["local_id"], "") else ""
            it.update(key=key, seconds=round((m.get("voice_ms") or 0) / 1000, 1),
                      mandarin=(r and r.get("text")) or old or None, stt_error=r and r.get("error"),
                      candidates=(r or {}).get("candidates") or {}, local=drafts.get(key, {}),
                      label=lab and {"wu": lab.get("wu"), "meaning": lab.get("meaning"), "status": lab.get("status")})
        if m["type"] == "image" and m.get("image_path"):
            it["image"] = os.path.basename(m["image_path"])
        if m["type"] == "app":
            it["text"] = m.get("text") or m.get("title") or "[链接/文件]"
        items.append(it)
        cmd = _command(m, on_words, off_words)
        if cmd:
            switches.append({"kind": "switch", "time": m["time"], "on": cmd == "on",
                             "by": m.get("sender_name") or m.get("sender")})
    if con:
        for t, role, text in con.execute("select time, role, text from messages where chat = ? and role = 'assistant'",
                                         (chat_id,)):
            items.append({"kind": "msg", "time": t, "type": "text", "sender": "莫名其妙（机器人）", "bot": True,
                          "text": text})
        row = _switch_row(con, chat_id)
        if row and not any(abs(s["time"] - row[1]) < 5 for s in switches):  # set outside a chat command
            switches.append({"kind": "switch", "time": row[1], "on": bool(row[0]), "by": "（直接设置）"})
    switches.sort(key=lambda s: s["time"])
    # state periods for the strip: from the first message, default state, then each switch
    start = min([i["time"] for i in items] + [s["time"] for s in switches], default=0)
    state = _default_on(cfg, chat_id, name) if not switches else not switches[0]["on"]
    periods, t0 = [], start
    for s in switches:
        if s["on"] != state:
            periods.append({"from": t0, "to": s["time"], "on": state})
            t0, state = s["time"], s["on"]
    row = con and _switch_row(con, chat_id)
    current = bool(row[0]) if row else state
    periods.append({"from": t0, "to": None, "on": state})
    timeline = sorted(items + switches, key=lambda x: (x["time"], x["kind"] != "switch"))
    return {"chat": chat_id, "name": name, "is_group": is_group, "on": current, "periods": periods,
            "items": timeline}
