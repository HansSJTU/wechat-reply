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
import wave

from stt.paths import DATA, GLOSSARY, LABELS as LABEL_DIR, WU_TOKENS as TOKENS

MESSAGES = os.path.join(DATA, "messages.jsonl")
RESULTS = os.path.join(DATA, "stt_results.jsonl")
VOICE_STT = os.path.join(DATA, "voice_stt.jsonl")
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
DRAFTERS = ["whisper-wu", "wu-local", "wu-ft", "whisper-ft"]
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
    return _load_json(DRAFTS, {})


def labels() -> dict:
    return {r["key"]: r for r in _jsonl(LABELS)}


def bot_stt() -> dict:
    """What the bot transcribed while a chat was 开启 (data/voice_stt.jsonl): key -> record (latest)."""
    return {r["key"]: r for r in _jsonl(VOICE_STT)}


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


def draft(keys: list[str] | None = None, log=print) -> int:
    """Fill drafts for voices that have none (or only `keys`). Loads the local models once."""
    from stt.audio import Audio
    from stt.core import PROVIDERS
    with DRAFT_LOCK:
        ready = [n for n in DRAFTERS if PROVIDERS[n].ready()]
        if not ready:
            return 0
        d = drafts()
        todo = [v for v in voices() if (keys is None or v["key"] in keys)
                and any(n not in d.get(v["key"], {}) for n in ready)][::-1]  # page order: newest / top first
        for v in todo:
            a = Audio(v["wav"])
            got = d.setdefault(v["key"], {})
            for n in ready:
                if n in got:
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
            log(f"drafted {v['key']}: {got.get(ready[0], '')}")
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
