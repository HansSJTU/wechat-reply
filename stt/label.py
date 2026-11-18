#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "websockets>=13", "sherpa-onnx", "torch", "openai-whisper"]
# ///
"""Label saved WeChat voice messages with what was actually said in Wu, for fine-tuning wu-local.

  uv run stt/label.py serve      local page at http://127.0.0.1:8765 (drafts new voices in the background)
  uv run stt/label.py draft      pre-fill drafts for every voice message that has none
  uv run stt/label.py export     data/labels/wenet/{train,test}.jsonl for WeNet ({"key","wav","txt"})
  uv run stt/label.py convert "我们这个鱼要吃吗 要吃的我拿一点起lang" [--key 12_voice.wav]
                                 rough mixed transcription → 余姚话 (说法表 spellings) + 普通话 via Gemini

Voices come from data/messages.jsonl (what monitor.py saves). Drafts from whisper-wu and wu-local go to
data/labels/drafts.json; labels are appended to data/labels/labels.jsonl (latest line per key wins).
A fixed ~20% of clips (by hash of the file name) is the test split and must never be trained on.
Everything stays on this machine; the server only listens on 127.0.0.1.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MESSAGES = os.path.join(ROOT, "data", "messages.jsonl")
RESULTS = os.path.join(ROOT, "data", "stt_results.jsonl")
LABEL_DIR = os.path.join(ROOT, "data", "labels")
DRAFTS = os.path.join(LABEL_DIR, "drafts.json")
LABELS = os.path.join(LABEL_DIR, "labels.jsonl")
EXPORT_DIR = os.path.join(LABEL_DIR, "wenet")
GLOSSARY = os.path.join(LABEL_DIR, "glossary.json")
# 说法表: one chosen spelling for each 余姚话 word, its Mandarin meaning, and spellings to
# flag (draft habits from 上海话 like 伊/今朝, or homophones like 几么). Keeps labels consistent and
# doubles as a dictionary for the LLM. Spellings follow the 余姚城区话 tense article (今末/上末/渠/
# 首西/葛/噎/啷哉); the full list lives in glossary.json, this seed only applies when it is missing.
GLOSSARY_SEED = [
    {"wu": "今末", "mandarin": "今天", "avoid": ["几末", "几米", "几未", "几么", "今朝", "起末", "去末"]},
    {"wu": "杂话", "mandarin": "怎么样/怎么说", "avoid": ["咋话", "咋窝"]},
    {"wu": "姆妈", "mandarin": "妈妈", "avoid": ["么吗"]},
    {"wu": "渠", "mandarin": "他/她", "avoid": ["伊"]},
]
TOKENS = os.path.join(ROOT, "models", "wu-u2pp-int8", "tokens.txt")
UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_ui.html")
DRAFTERS = ["whisper-wu", "wu-local"]
# done: exported to its split · skip: unclear · bad: broken audio · exclude: labeled or not, kept out of
# both train and test on purpose (private, a test message to the bot, not representative …)
STATUSES = ("done", "skip", "bad", "exclude")
PUNCT = re.compile(r"[\s　-〿＀-／：-＠［-｀｛-･.,!?;:'\"()\[\]…—-]+")

_lock = threading.Lock()


def split_of(key: str) -> str:
    return "test" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 5 == 0 else "train"


def voices() -> list[dict]:
    """Every saved voice message whose audio file exists, oldest first."""
    out, seen = [], set()
    if os.path.exists(MESSAGES):
        for line in open(MESSAGES, encoding="utf-8"):
            m = json.loads(line)
            p = m.get("voice_path")
            if m.get("type") != "voice" or not p or not os.path.exists(p):
                continue
            key = os.path.basename(p)
            if key in seen:
                continue
            seen.add(key)
            out.append({"key": key, "wav": p, "seconds": round((m.get("voice_ms") or 0) / 1000, 1),
                        "time": m.get("time"), "chat": m.get("chat_name") or m.get("chat"),
                        "sender": m.get("sender_name") or m.get("sender"), "split": split_of(key)})
    return out


def _load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def drafts() -> dict:
    return _load_json(DRAFTS, {})


def labels() -> dict:
    out = {}
    if os.path.exists(LABELS):
        for line in open(LABELS, encoding="utf-8"):
            r = json.loads(line)
            out[r["key"]] = r
    return out


def meanings() -> dict:
    """Mandarin glosses given earlier as CLI references (-e), keyed by file name."""
    out = {}
    if os.path.exists(RESULTS):
        for line in open(RESULTS, encoding="utf-8"):
            r = json.loads(line)
            if r.get("expect"):
                out[r["file"]] = r["expect"]
    return out


def glossary() -> list[dict]:
    g = _load_json(GLOSSARY, None)
    if g is None:
        save_glossary(GLOSSARY_SEED)
        g = GLOSSARY_SEED
    return g


def save_glossary(entries: list[dict]) -> list[dict]:
    clean = []
    for e in entries:
        wu = str(e.get("wu", "")).strip()
        if not wu:
            continue
        avoid = e.get("avoid", [])
        if isinstance(avoid, str):
            avoid = re.split(r"[,，、\s]+", avoid)
        clean.append({"wu": wu, "mandarin": str(e.get("mandarin", "")).strip(),
                      "avoid": [a.strip() for a in avoid if a.strip() and a.strip() != wu]})
    os.makedirs(LABEL_DIR, exist_ok=True)
    with open(GLOSSARY + ".tmp", "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=1)
    os.replace(GLOSSARY + ".tmp", GLOSSARY)
    return clean


def vocab() -> set:
    if not os.path.exists(TOKENS):
        return set()
    return {ln.split()[0] for ln in open(TOKENS, encoding="utf-8") if ln.strip()}


def draft(keys: list[str] | None = None, log=print) -> int:
    """Fill drafts for voices that have none (or only `keys`). Loads the local models once."""
    from stt.audio import Audio
    from stt.core import PROVIDERS
    with _lock:
        d = drafts()
        todo = [v for v in voices() if (keys is None or v["key"] in keys)
                and any(n not in d.get(v["key"], {}) for n in DRAFTERS)]
        for v in todo:
            a = Audio(v["wav"])
            got = d.setdefault(v["key"], {})
            for n in DRAFTERS:
                if n in got or not PROVIDERS[n].ready():
                    continue
                try:
                    got[n] = PROVIDERS[n].run(a)
                except Exception as e:
                    got[n] = ""
                    log(f"{v['key']} {n}: {type(e).__name__}: {e}")
            os.makedirs(LABEL_DIR, exist_ok=True)
            with open(DRAFTS + ".tmp", "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=1)
            os.replace(DRAFTS + ".tmp", DRAFTS)
            log(f"drafted {v['key']}: {got.get(DRAFTERS[0], '')}")
        return len(todo)


CONVERT_MODEL = "gemini-4-hydrogen"
# Choices on the labeling page (model id -> label); the first is the default.
CONVERT_MODELS = {"gemini-4-hydrogen": "Gemini hydrogen", "gpt-6.2-sol": "GPT-6.2 sol"}
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
    import base64
    import httpx
    from stt import keys
    if model not in CONVERT_MODELS:
        raise ValueError(f"unknown model {model}")
    gemini = model.startswith("gemini")
    gl = "\n".join(f"{e['wu']} = {e['mandarin']}" + (f"；别写成：{'、'.join(e['avoid'])}" if e.get("avoid") else "")
                   for e in glossary())
    v = next((v for v in voices() if v["key"] == key), None) if key else None
    audio = bool(v and with_audio and gemini)
    prompt = CONVERT_PROMPT.format(rough=rough.strip(), glossary=gl)
    if not audio:  # text only: don't let it spend the notes on "no audio was given"
        prompt = prompt.replace("请结合录音和标注者的句子", "这次没有附录音，只根据标注者的句子").replace(
            "；只有录音明显不同时才改，并在 notes 里说明", "").replace("- notes：不确定的地方", "- notes：用词上不确定的地方（不用提没有录音）")
    fields = ("wu", "mandarin", "notes")
    t = time.time()
    if gemini:
        parts = [{"inline_data": {"mime_type": "audio/wav",
                                  "data": base64.b64encode(open(v["wav"], "rb").read()).decode()}}] if audio else []
        parts.append({"text": prompt})
        schema = {"type": "OBJECT", "properties": {k: {"type": "STRING"} for k in fields}, "required": list(fields)}
        r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                       headers={"x-goog-api-key": keys.get("gemini_api_key")}, timeout=180,
                       json={"contents": [{"role": "user", "parts": parts}],
                             "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema}})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {' '.join(r.text.split())[:300]}")
        cands = r.json().get("candidates") or []
        text = "".join(p.get("text", "") for p in (cands[0].get("content", {}).get("parts", []) if cands else [])
                       if not p.get("thought"))
    else:
        schema = {"type": "object", "properties": {k: {"type": "string"} for k in fields},
                  "required": list(fields), "additionalProperties": False}
        r = httpx.post("https://api.openai.com/v1/chat/completions",
                       headers={"Authorization": f"Bearer {keys.get('openai_api_key')}"}, timeout=180,
                       json={"model": model, "messages": [{"role": "user", "content": prompt}],
                             "response_format": {"type": "json_schema",
                                                 "json_schema": {"name": "label", "strict": True, "schema": schema}}})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {' '.join(r.text.split())[:300]}")
        text = r.json()["choices"][0]["message"]["content"]
    out = json.loads(text)
    return {k: str(out.get(k, "")).strip() for k in fields} | {
        "seconds": round(time.time() - t, 1), "model": model, "audio": audio}


def normalize(text: str) -> str:
    return PUNCT.sub("", text)


def export() -> dict:
    os.makedirs(EXPORT_DIR, exist_ok=True)
    lab, voc = labels(), vocab()
    by_key = {v["key"]: v for v in voices()}
    counts = {"train": [0, 0.0], "test": [0, 0.0]}
    oov = set()
    files = {s: open(os.path.join(EXPORT_DIR, f"{s}.jsonl"), "w", encoding="utf-8") for s in counts}
    for key, r in sorted(lab.items()):
        v = by_key.get(key)
        txt = normalize(r.get("wu", ""))
        if r.get("status") != "done" or not v or not txt:
            continue
        oov |= {c for c in txt if voc and c not in voc}
        s = v["split"]
        files[s].write(json.dumps({"key": key.removesuffix(".wav"), "wav": v["wav"], "txt": txt},
                                  ensure_ascii=False) + "\n")
        counts[s][0] += 1
        counts[s][1] += v["seconds"]
    for f in files.values():
        f.close()
    return {"dir": EXPORT_DIR, **{s: {"clips": n, "minutes": round(sec / 60, 1)} for s, (n, sec) in counts.items()},
            "chars_not_in_model_vocab": "".join(sorted(oov))}


# ---------------------------------------------------------------- web server

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, open(UI, "rb").read(), "text/html; charset=utf-8")
        elif self.path == "/api/items":
            d, lab, mean = drafts(), labels(), meanings()
            items = [{**v, "drafts": d.get(v["key"], {}), "label": lab.get(v["key"]),
                      "meaning_ref": mean.get(v["key"], "")} for v in voices()]
            self._json({"items": items, "drafters": DRAFTERS, "drafting": _lock.locked()})
        elif self.path == "/api/vocab":
            self._json({"chars": "".join(sorted(vocab()))})
        elif self.path == "/api/glossary":
            self._json({"entries": glossary()})
        elif self.path == "/api/models":
            self._json({"models": [{"id": k, "name": n} for k, n in CONVERT_MODELS.items()]})
        elif self.path.startswith("/audio/"):
            key = self.path.removeprefix("/audio/")
            v = next((v for v in voices() if v["key"] == key), None)  # only files listed in messages.jsonl
            if not v:
                return self._send(404, b"not found", "text/plain")
            self._send(200, open(v["wav"], "rb").read(), "audio/wav")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/api/label", "/api/glossary", "/api/convert"):
            return self._send(404, b"not found", "text/plain")
        r = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
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
               "status": r["status"], "split": split_of(r["key"]), "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        os.makedirs(LABEL_DIR, exist_ok=True)
        with open(LABELS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._json({"ok": True, "label": rec})


def serve(port: int, auto_draft: bool):
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
    c = sub.add_parser("convert", help="rough mixed transcription → 余姚话 + 普通话 via Gemini or GPT")
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
