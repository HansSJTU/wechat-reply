#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "websockets>=13", "sherpa-onnx", "torch", "openai-whisper"]
# ///
"""Label saved WeChat voice messages with what was actually said in Wu, for fine-tuning wu-local.

  uv run stt/label.py serve      local page at http://127.0.0.1:8765 (drafts new voices in the background)
  uv run stt/label.py draft      pre-fill drafts for every voice message that has none
  uv run stt/label.py export     data/labels/wenet/{train,test}.jsonl for WeNet ({"key","wav","txt"})

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
# doubles as a dictionary for the LLM. Seeded with the spellings settled on 2026-11-15.
GLOSSARY_SEED = [
    {"wu": "今末", "mandarin": "今天", "avoid": ["几么", "今朝", "起末", "去末"]},
    {"wu": "咋话", "mandarin": "怎么样", "avoid": ["咋窝"]},
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
        elif self.path.startswith("/audio/"):
            key = self.path.removeprefix("/audio/")
            v = next((v for v in voices() if v["key"] == key), None)  # only files listed in messages.jsonl
            if not v:
                return self._send(404, b"not found", "text/plain")
            self._send(200, open(v["wav"], "rb").read(), "audio/wav")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/api/label", "/api/glossary"):
            return self._send(404, b"not found", "text/plain")
        r = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
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
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(a.port, not a.no_draft)
    elif a.cmd == "draft":
        print(f"drafted {draft()} clips")
    else:
        print(json.dumps(export(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
