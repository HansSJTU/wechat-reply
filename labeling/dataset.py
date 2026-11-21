"""The labeling data: which clips exist, their drafts and labels, the 说法表, and the WeNet export.

Voices come from data/messages.jsonl (what monitor.py saves) plus data/labels/extra_clips*.jsonl (clips cut
from other sources by labeling/sources/). Drafts from the local models go to data/labels/drafts.json; labels
are appended to data/labels/labels.jsonl (latest line per key wins). A fixed ~20% of clips (by hash of the
file name) is the test split and must never be trained on. Stdlib only; the models load only in `draft()`.
"""
import glob
import hashlib
import json
import os
import re
import threading
import time
import wave

from stt.paths import DATA, GLOSSARY, LABELS as LABEL_DIR, WU_TOKENS as TOKENS

MESSAGES = os.path.join(DATA, "messages.jsonl")
RESULTS = os.path.join(DATA, "stt_results.jsonl")
VOICE_STT = os.path.join(DATA, "voice_stt.jsonl")
VOICE_LOCAL = os.path.join(DATA, "voice_local.jsonl")  # 关闭 chats: local models only
DRAFTS = os.path.join(LABEL_DIR, "drafts.json")
LABELS = os.path.join(LABEL_DIR, "labels.jsonl")
EXPORT_DIR = os.path.join(LABEL_DIR, "wenet")
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
# Clips from other sources (labeling/sources/): one JSON line each with key, wav, seconds,
# source, sender, split (always train), prompt (on-screen Mandarin) and prefill (proposed 余姚话).
EXTRA = os.path.join(LABEL_DIR, "extra_clips.jsonl")
# Local ASR drafts for the labeling page, best first (the first one prefills the box). The released base
# models (whisper-wu, wu-local) were dropped 2026-11-20: older clips still carry their drafts in
# drafts.json, which shown_drafts() leaves out.
DRAFTERS = ["whisper-ft", "wu-ft"]
# done: exported to its split · skip: unclear · bad: broken audio · exclude: labeled or not, kept out of
# both train and test on purpose (private, a test message to the bot, not representative …)
STATUSES = ("done", "skip", "bad", "exclude")
PUNCT = re.compile(r"[\s　-〿＀-／：-＠［-｀｛-･.,!?;:'\"()\[\]…—-]+")

DRAFT_LOCK = threading.Lock()     # held while draft() runs (the page shows "drafting")
GLOSSARY_LOCK = threading.Lock()  # serializes glossary.json.tmp writes across HTTP threads
LABELS_LOCK = threading.Lock()    # serializes appends to labels.jsonl across HTTP threads


def _jsonl(path: str) -> list[dict]:
    """Read a JSONL file, skipping blank or half-written lines (e.g. while monitor.py appends)."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def split_of(key: str) -> str:
    return "test" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 5 == 0 else "train"


def voices() -> list[dict]:
    """Every saved voice message whose audio file exists, oldest first."""
    out, seen = [], set()
    for m in _jsonl(MESSAGES):
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
    out = {}
    for path in sorted(glob.glob(os.path.join(LABEL_DIR, "extra_clips*.jsonl"))):
        for e in _jsonl(path):
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
    """Everything in drafts.json (draft() rewrites the whole file, so nothing in it is dropped)."""
    return _load_json(DRAFTS, {})


def shown_drafts() -> dict:
    """drafts() limited to the current DRAFTERS: what the pages and the clip builders use."""
    return {k: {n: t for n, t in v.items() if n in DRAFTERS} for k, v in drafts().items()}


def labels() -> dict:
    return {r["key"]: r for r in _jsonl(LABELS)}


def bot_stt() -> dict:
    """What the bot transcribed while a chat was 开启 (data/voice_stt.jsonl): key -> record (latest)."""
    return {r["key"]: r for r in _jsonl(VOICE_STT)}


def bot_heard() -> dict:
    """Every voice the bot ran its local models on: 开启 (voice_stt.jsonl) and 关闭 (voice_local.jsonl,
    whisper-ft / wu-ft only). key -> record with "candidates"; 开启 wins if both exist."""
    return {**{r["key"]: r for r in _jsonl(VOICE_LOCAL)}, **bot_stt()}


def meanings() -> dict:
    """Mandarin glosses given earlier as CLI references (-e), keyed by file name."""
    return {r["file"]: r["expect"] for r in _jsonl(RESULTS) if r.get("expect")}


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
    with GLOSSARY_LOCK:
        os.makedirs(LABEL_DIR, exist_ok=True)
        with open(GLOSSARY + ".tmp", "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=1)
        os.replace(GLOSSARY + ".tmp", GLOSSARY)
    return clean


def glossary_prompt() -> str:
    """The 说法表 as prompt lines: "写法 = 普通话意思；别写成：…"."""
    return "\n".join(f"{e['wu']} = {e['mandarin']}" + (f"；别写成：{'、'.join(e['avoid'])}" if e.get("avoid") else "")
                     for e in glossary())


def vocab() -> set:
    if not os.path.exists(TOKENS):
        return set()
    with open(TOKENS, encoding="utf-8") as f:
        return {ln.split()[0] for ln in f if ln.strip()}


BOT_GRACE = 120  # s: the bot transcribes a new voice in a 开启 chat within seconds; wait for that first


def _bot_text(rec, name):
    """The bot's own result of local model `name` for this voice (its "candidates"), if usable."""
    t = ((rec or {}).get("candidates") or {}).get(name)
    return t if t and not t.startswith("✗") else None


def _release_models():
    from stt import local_whisper_wu, local_wu
    local_whisper_wu.unload()
    local_wu.unload()


def draft(keys: list[str] | None = None, log=print, now: float | None = None) -> int:
    """Fill drafts for voices that have none (or only `keys`).

    The bot already runs the same models (whisper-ft, wu-ft) on every voice it receives, 开启 or 关闭:
    those results are taken from voice_stt.jsonl / voice_local.jsonl as they are. The models run here only
    for the rest (voices from before those files, a model that failed in the bot, clips from other sources)
    and are released afterwards, so the labeling server does not hold a second copy of Whisper."""
    from stt.audio import Audio
    from stt.core import PROVIDERS
    with DRAFT_LOCK:
        now = time.time() if now is None else now
        ready = {n for n in DRAFTERS if PROVIDERS[n].ready()}
        d, bot = drafts(), bot_heard()

        def missing(v):
            have, rec = d.get(v["key"], {}), bot.get(v["key"])
            return [n for n in DRAFTERS if n not in have and (n in ready or _bot_text(rec, n))]

        def waiting(v):  # a new WeChat voice the bot is likely transcribing right now
            return v.get("source") == "wechat" and v["key"] not in bot and now - (v.get("time") or 0) < BOT_GRACE

        todo = [v for v in voices() if (keys is None or v["key"] in keys) and missing(v) and not waiting(v)][::-1]
        ran = False  # page order: newest / top first
        try:
            for v in todo:
                names, rec, audio = missing(v), bot.get(v["key"]), None
                got = d.setdefault(v["key"], {})
                for n in names:
                    if (t := _bot_text(rec, n)) is not None:
                        got[n] = t
                        continue
                    audio = audio or Audio(v["wav"])
                    ran = True
                    try:
                        got[n] = PROVIDERS[n].run(audio)
                    except Exception as e:
                        got[n] = ""
                        log(f"{v['key']} {n}: {type(e).__name__}: {e}")
                os.makedirs(LABEL_DIR, exist_ok=True)
                with open(DRAFTS + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(d, f, ensure_ascii=False, indent=1)
                os.replace(DRAFTS + ".tmp", DRAFTS)
                src = "" if audio else " (from the bot)"
                log(f"drafted {v['key']}{src}: {got.get(DRAFTERS[0], '')}")
        finally:
            if ran:
                _release_models()
        return len(todo)


def normalize(text: str) -> str:
    return PUNCT.sub("", text)


def cut_wav(src: str, dst: str, start: float, end: float) -> float:
    """Copy [start, end) seconds of a PCM wav to dst; returns the kept length in seconds."""
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
