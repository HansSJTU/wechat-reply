#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "sherpa-onnx", "torch", "openai-whisper"]
# ///
"""Label saved WeChat voice messages with what was actually said in Wu, for fine-tuning wu-local.

  uv run labeling/label.py serve    local page at http://127.0.0.1:8765 (drafts new voices in the background;
                                   /chats is the chat viewer)
  uv run labeling/label.py draft    pre-fill drafts for every voice message that has none
  uv run labeling/label.py export   data/labels/wenet/{train,test}.jsonl for WeNet ({"key","wav","txt"})
  uv run labeling/label.py convert "我们这个鱼要吃吗 要吃的我拿一点起lang" [--key 12_voice.wav]
                                   rough mixed transcription → 余姚话 (说法表 spellings) + 普通话

Data (clips, drafts, labels, 说法表, export): labeling/dataset.py.
Everything stays on this machine; the server only listens on 127.0.0.1.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labeling import chat_view  # noqa: E402
from labeling.dataset import (DRAFTERS, LABEL_DIR, LABELS, LABELS_LOCK, STATUSES, DRAFT_LOCK, bot_stt,  # noqa: E402
                              draft, export, glossary, glossary_prompt, labels, meanings, save_glossary,
                              shown_drafts, split_for, vocab, voices)
from labeling.llm import llm_json  # noqa: E402
from stt.paths import DATA  # noqa: E402
from urllib.parse import parse_qs, unquote, urlparse  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

CONVERT_MODEL = "gpt-6.2-sol"
# Choices on the labeling page (model id -> label); the first is the default. Gemini only when picked by hand.
CONVERT_MODELS = {"gpt-6.2-sol": "GPT-6.2 sol", "agy:Gemini 4 Neon (High)": "Gemini 4 Neon (agy)",
                  "gemini-4-hydrogen": "Gemini hydrogen"}
CONVERT_PROMPT = """你在帮人标注余姚话（浙江余姚的吴语）微信语音，用来训练语音识别模型。
标注者听了录音，写下了他听到的句子。这句话是混合的：有吴语字、有普通话意思、也有拼音读音（如 lang、o sei、sa ning）。

标注者听到的：{rough}

请结合录音和标注者的句子，输出：
- wu：说话人实际说的余姚话，逐字照读音写。
  · 说法表里有的词一律用说法表的写法（看“别写成”，那些写法都要换掉）。
  · 拼音换成说法表的写法；说法表没有的，用读音最接近的常用汉字。
  · 普通话意思要换回说话人实际说的吴语说法（如 你→侬、不→勿、的→葛、拿→驮、我们→阿拉、这个→噎个、吗→伐、了→哉），但说话人本来就说普通话的词照普通话写。
  · 重复、口吃、语气词照说的写，不要删，不要加。人名、专有名词照写。
  · 以标注者听到的为准；只有录音明显不同时才改，并在 notes 里说明。
- mandarin：这句话的普通话意思，自然通顺，带标点。
- notes：不确定的地方，一两句话，中文；没有就留空。

说法表（写法 = 普通话意思；别写成：…）：
{glossary}"""


def convert(key: str | None, rough: str, with_audio: bool = True, model: str = CONVERT_MODEL) -> dict:
    """Ask an LLM to turn a rough mixed transcription into 余姚话 (说法表 spellings) + Mandarin.
    Only Gemini gets the audio; GPT models work from the text alone."""
    if model not in CONVERT_MODELS:
        raise ValueError(f"unknown model {model}")
    gemini = model.startswith("gemini")
    v = next((v for v in voices() if v["key"] == key), None) if key else None
    audio = bool(v and with_audio and gemini)
    prompt = CONVERT_PROMPT.format(rough=rough.strip(), glossary=glossary_prompt())
    if not audio:  # text only: don't let it spend the notes on "no audio was given"
        prompt = prompt.replace("请结合录音和标注者的句子", "这次没有附录音，只根据标注者的句子").replace(
            "；只有录音明显不同时才改，并在 notes 里说明", "").replace("- notes：不确定的地方", "- notes：用词上不确定的地方（不用提没有录音）")
    fields = ("wu", "mandarin", "notes")
    t = time.time()
    out = llm_json(model, prompt, fields, v["wav"] if audio else None)
    return out | {"seconds": round(time.time() - t, 1), "model": model, "audio": audio}


# ---------------------------------------------------------------- web server

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype: str, headers: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "label_ui.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/items":
            d, lab, mean, bot = shown_drafts(), labels(), meanings(), bot_stt()

            def cloud(key):  # the bot's cloud candidates (local ones are drafted here anyway)
                c = (bot.get(key) or {}).get("candidates") or {}
                return {f"云·{n}": t for n, t in c.items() if n not in DRAFTERS and t and not t.startswith("✗")}

            items = [{**v, "drafts": {**({"字幕": v["prefill"]} if v.get("prefill") else {}), **d.get(v["key"], {}),
                                      **cloud(v["key"])},
                      "label": lab.get(v["key"]),
                      "meaning_ref": mean.get(v["key"]) or (bot.get(v["key"]) or {}).get("text") or v.get("prompt", "")}
                     for v in voices()]
            self._json({"items": items, "drafters": DRAFTERS, "drafting": DRAFT_LOCK.locked()})
        elif path in ("/chats", "/chats/"):  # chat viewer (labeling/chat_view.py)
            with open(os.path.join(HERE, "chats_ui.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/chats":
            self._json({"chats": chat_view.chats()})
        elif path == "/api/chat":
            self._json(chat_view.chat(parse_qs(parsed.query).get("c", [""])[0]))
        elif path.startswith("/media/"):  # images saved by the bot (data/media only)
            name = os.path.basename(unquote(path.removeprefix("/media/")))
            fpath = os.path.join(DATA, "media", name)
            if not os.path.isfile(fpath) or not name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                return self._send(404, b"not found", "text/plain")
            ext = name.rsplit(".", 1)[1].lower()
            with open(fpath, "rb") as f:
                self._send(200, f.read(), "image/" + ("jpeg" if ext == "jpg" else ext))
        elif path == "/api/vocab":
            self._json({"chars": "".join(sorted(vocab()))})
        elif path == "/api/glossary":
            self._json({"entries": glossary()})
        elif path == "/api/models":
            self._json({"models": [{"id": k, "name": n} for k, n in CONVERT_MODELS.items()]})
        elif path.startswith("/audio/"):
            key = unquote(path.removeprefix("/audio/"))
            v = next((v for v in voices() if v["key"] == key), None)  # only files listed in messages.jsonl
            if not v:
                return self._send(404, b"not found", "text/plain")
            with open(v["wav"], "rb") as f:
                data = f.read()
            # Range support: without it the browser can't seek (playing from a point on the waveform)
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
            if m and (m[1] or m[2]):
                a, b = (int(m[1]), int(m[2] or len(data) - 1)) if m[1] else (max(0, len(data) - int(m[2])), len(data) - 1)
                b = min(b, len(data) - 1)
                if a > b:
                    return self._send(416, b"", "audio/wav", {"Content-Range": f"bytes */{len(data)}"})
                return self._send(206, data[a:b + 1], "audio/wav",
                                  {"Content-Range": f"bytes {a}-{b}/{len(data)}", "Accept-Ranges": "bytes"})
            self._send(200, data, "audio/wav", {"Accept-Ranges": "bytes"})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/api/label", "/api/glossary", "/api/convert"):
            return self._send(404, b"not found", "text/plain")
        try:
            length = int(self.headers.get("Content-Length", 0))
            r = json.loads(self.rfile.read(length))
            if not isinstance(r, dict):
                raise ValueError("JSON object expected")
        except (ValueError, json.JSONDecodeError):
            return self._json({"error": "invalid json"}, 400)
        if self.path == "/api/convert":
            if not str(r.get("rough", "")).strip():
                return self._json({"error": "empty"}, 400)
            try:
                return self._json({"ok": True, **convert(r.get("key"), r["rough"], r.get("audio", True),
                                                          r.get("model") or CONVERT_MODEL)})
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 502)
        if self.path == "/api/glossary":
            if not isinstance(r.get("entries"), list):
                return self._json({"error": "bad request"}, 400)
            return self._json({"ok": True, "entries": save_glossary(r["entries"])})
        if r.get("key") not in {v["key"] for v in voices()} or r.get("status") not in STATUSES:
            return self._json({"error": "bad request"}, 400)
        rec = {"key": r["key"], "wu": r.get("wu", "").strip(), "meaning": r.get("meaning", "").strip(),
               "status": r["status"], "split": split_for(r["key"]), "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        t = r.get("trim")
        if (isinstance(t, list) and len(t) == 2 and all(isinstance(x, (int, float)) for x in t)
                and 0 <= t[0] < t[1]):
            rec["trim"] = [round(t[0], 3), round(t[1], 3)]
        with LABELS_LOCK:
            os.makedirs(LABEL_DIR, exist_ok=True)
            with open(LABELS, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._json({"ok": True, "label": rec})


def serve(port: int, auto_draft: bool):
    from wxbot import housekeeping
    from wxbot.config import load_config
    housekeeping.start(load_config(), media=False)  # rotates data/label.out (media is the bot's job)
    if auto_draft:
        def loop():
            while True:
                try:
                    draft(log=lambda s: print("  " + s, flush=True))
                except Exception as e:
                    print(f"  drafting failed: {type(e).__name__}: {e}", flush=True)
                time.sleep(30)  # pick up voices monitor.py saves while the page is open
        threading.Thread(target=loop, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"labeling page: http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-draft", action="store_true", help="don't draft new voices in the background")
    sub.add_parser("draft")
    sub.add_parser("export")
    c = sub.add_parser("convert", help="rough mixed transcription → 余姚话 + 普通话 via GPT (or Gemini)")
    c.add_argument("rough")
    c.add_argument("--key", help="voice file name, to send its audio too (Gemini only)")
    c.add_argument("--model", default=CONVERT_MODEL, choices=list(CONVERT_MODELS))
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(a.port, not a.no_draft)
    elif a.cmd == "convert":
        print(json.dumps(convert(a.key, a.rough, model=a.model), ensure_ascii=False, indent=1))
    elif a.cmd == "draft":
        print(f"drafted {draft()} clips")
    else:
        print(json.dumps(export(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
