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
# Clips from other sources (cut from videos by stt/bili_clips.py): one JSON line each with key, wav, seconds,
# source, sender, split (always train), prompt (on-screen Mandarin) and prefill (proposed 余姚话).
EXTRA = os.path.join(LABEL_DIR, "extra_clips.jsonl")
UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_ui.html")
DRAFTERS = ["whisper-wu", "wu-local", "wu-ft", "whisper-ft"]
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
                        "sender": m.get("sender_name") or m.get("sender"), "split": split_of(key),
                        "source": "wechat"})
    return out + [e for e in extras().values() if e["key"] not in seen and os.path.exists(e["wav"])]


def extras() -> dict:
    """Clips from every data/labels/extra_clips*.jsonl (videos with on-screen text, unlabeled web audio, …)."""
    import glob
    out = {}
    for path in sorted(glob.glob(os.path.join(LABEL_DIR, "extra_clips*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                e = json.loads(line)
                out[e["key"]] = {"split": "train", **e}
    return out


def split_for(key: str) -> str:
    e = extras().get(key)
    return e["split"] if e else split_of(key)


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


def bot_stt() -> dict:
    """What the bot transcribed while a chat was 开启 (data/voice_stt.jsonl): key -> record (latest)."""
    out = {}
    path = os.path.join(ROOT, "data", "voice_stt.jsonl")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
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
                and any(n not in d.get(v["key"], {}) for n in DRAFTERS)][::-1]  # page order: newest / top first
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
    out = llm_json(model, prompt, fields, v["wav"] if audio else None)
    return out | {"seconds": round(time.time() - t, 1), "model": model, "audio": audio}


def llm_json(model: str, prompt: str, fields: tuple[str, ...], wav: str | None = None) -> dict:
    """One structured LLM call returning the string `fields`. Gemini can also get a WAV clip; GPT is text only.
    "agy:<model name>" runs the Antigravity CLI instead (`agy -p … --model …`, text only, no API key)."""
    import base64
    import httpx
    from stt import keys
    if model.startswith("agy:"):
        return _agy_json(model.removeprefix("agy:"), prompt, fields, wav)
    if model.startswith("gemini"):
        parts = [{"inline_data": {"mime_type": "audio/wav",
                                  "data": base64.b64encode(open(wav, "rb").read()).decode()}}] if wav else []
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
        if wav:
            raise ValueError(f"{model} can't take audio")
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
    return {k: str(out.get(k, "")).strip() for k in fields}


def _agy_json(model: str, prompt: str, fields: tuple[str, ...], wav: str | None) -> dict:
    import subprocess
    import tempfile
    if wav:
        raise ValueError("agy is used text only")
    schema = {"type": "object", "properties": {k: {"type": "string"} for k in fields}, "required": list(fields)}
    with tempfile.TemporaryDirectory() as d:  # empty workspace: nothing for the agent to read
        r = subprocess.run(["agy", "-p", prompt, "--model", model, "--json-schema", json.dumps(schema),
                            "--output-format", "json", "--print-timeout", "300s", "--sandbox"],
                           cwd=d, capture_output=True, text=True, timeout=360)
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"agy: {(r.stderr or r.stdout).strip()[:300]}")
    if out.get("status") != "SUCCESS" or not isinstance(out.get("structured_output"), dict):
        raise RuntimeError(f"agy: {out.get('status')} {str(out.get('response'))[:300]}")
    return {k: str(out["structured_output"].get(k, "")).strip() for k in fields}


def normalize(text: str) -> str:
    return PUNCT.sub("", text)


def cut_wav(src: str, dst: str, start: float, end: float) -> float:
    """Copy [start, end) seconds of a PCM wav to dst; returns the kept length in seconds."""
    import wave
    with wave.open(src, "rb") as w:
        rate, n = w.getframerate(), w.getnframes()
        a, b = max(0, int(start * rate)), min(n, int(end * rate))
        w.setpos(a)
        frames = w.readframes(max(0, b - a))
        params = w.getparams()
    with wave.open(dst + ".tmp", "wb") as o:
        o.setparams(params)
        o.writeframes(frames)
    os.replace(dst + ".tmp", dst)
    return (b - a) / rate


def export() -> dict:
    """Clips with a trim (首尾空白 cut on the page) are written trimmed to wenet/wav/; others point at the original."""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(os.path.join(EXPORT_DIR, "wav"), exist_ok=True)
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
        wav, sec = v["wav"], v["seconds"]
        if r.get("trim"):
            wav = os.path.join(EXPORT_DIR, "wav", key)
            sec = cut_wav(v["wav"], wav, *r["trim"])
        files[s].write(json.dumps({"key": key.removesuffix(".wav"), "wav": wav, "txt": txt},
                                  ensure_ascii=False) + "\n")
        counts[s][0] += 1
        counts[s][1] += sec
    for f in files.values():
        f.close()
    return {"dir": EXPORT_DIR, **{s: {"clips": n, "minutes": round(sec / 60, 1)} for s, (n, sec) in counts.items()},
            "chars_not_in_model_vocab": "".join(sorted(oov))}


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
        if self.path in ("/", "/index.html"):
            self._send(200, open(UI, "rb").read(), "text/html; charset=utf-8")
        elif self.path == "/api/items":
            d, lab, mean, bot = drafts(), labels(), meanings(), bot_stt()

            def cloud(key):  # the bot's cloud candidates (local ones are drafted here anyway)
                c = (bot.get(key) or {}).get("candidates") or {}
                return {f"云·{n}": t for n, t in c.items() if n not in DRAFTERS and t and not t.startswith("✗")}

            items = [{**v, "drafts": {**({"字幕": v["prefill"]} if v.get("prefill") else {}), **d.get(v["key"], {}),
                                      **cloud(v["key"])},
                      "label": lab.get(v["key"]),
                      "meaning_ref": mean.get(v["key"]) or (bot.get(v["key"]) or {}).get("text") or v.get("prompt", "")}
                     for v in voices()]
            self._json({"items": items, "drafters": DRAFTERS, "drafting": _lock.locked()})
        elif self.path in ("/chats", "/chats/"):  # chat viewer (stt/chat_view.py)
            self._send(200, open(os.path.join(os.path.dirname(UI), "chats_ui.html"), "rb").read(),
                       "text/html; charset=utf-8")
        elif self.path == "/api/chats":
            from stt import chat_view
            self._json({"chats": chat_view.chats()})
        elif self.path.startswith("/api/chat?"):
            from urllib.parse import parse_qs, urlparse
            from stt import chat_view
            self._json(chat_view.chat(parse_qs(urlparse(self.path).query).get("c", [""])[0]))
        elif self.path.startswith("/media/"):  # images saved by the bot (data/media only)
            name = os.path.basename(self.path.removeprefix("/media/"))
            path = os.path.join(ROOT, "data", "media", name)
            if not os.path.isfile(path) or not name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                return self._send(404, b"not found", "text/plain")
            ext = name.rsplit(".", 1)[1].lower()
            self._send(200, open(path, "rb").read(), "image/" + ("jpeg" if ext == "jpg" else ext))
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
            data = open(v["wav"], "rb").read()
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
               "status": r["status"], "split": split_for(r["key"]), "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        t = r.get("trim")
        if (isinstance(t, list) and len(t) == 2 and all(isinstance(x, (int, float)) for x in t)
                and 0 <= t[0] < t[1]):
            rec["trim"] = [round(t[0], 3), round(t[1], 3)]
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
