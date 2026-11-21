#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "numpy", "sherpa-onnx"]
# ///
"""Training clips from 余姚话 videos: on-screen text (bili_ocr.py) → timed clips → proposed 余姚话 →
the labeling page, where each clip is confirmed by ear like a WeChat voice.

  uv run labeling/sources/bili_clips.py build      group OCR segments into items, cut and trim 16 kHz clips
  uv run labeling/sources/bili_clips.py convert    propose 余姚话 for each item (GPT, text only; cached)
  uv run labeling/sources/bili_clips.py publish    write data/labels/extra_clips.jsonl for the labeling page

Media and OCR live in data/bili/ (<video>.m4a, ocr/<video>.jsonl); clips go to data/bili/clips/.
Kinds of on-screen text, set per source/time range in SOURCES:
  yubao    语保 grammar sentences: Mandarin + IPA + dialect characters → the dialect line, cleaned up
  wu       subtitles already in 余姚话 → cleaned up and normalized to 说法表 spellings
  prompt   a Mandarin prompt the speaker says in 余姚话 → inferred from the prompt + local ASR drafts
Train only, except ~20% of the 余姚 语保 sentences (hash of the key, like WeChat voices): a second test set.
"""
import argparse
import difflib
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from labeling.dataset import EXTRA, drafts as asr_drafts, glossary, glossary_prompt, split_of  # noqa: E402
from labeling.sources.wavio import RATE, read_pcm, speech_span, to_wav, write_wav  # noqa: E402
from stt.local_wu import FT_DIR  # noqa: E402
from stt.paths import DATA  # noqa: E402

DIR = os.path.join(DATA, "bili")
CLIPS = os.path.join(DIR, "clips")
ITEMS = os.path.join(DIR, "items.jsonl")
CONVERTED = os.path.join(DIR, "converted.json")
MODEL = "gpt-6.2-sol"  # default: text only. --model gemini-… also sends the clip (only when asked for)

# (video file stem, key prefix, who/what for the page, [(from s, to s, kind, lag)])
# lag: seconds the speech comes after the subtitle appears, found by matching the subtitles against a CTC
# character timeline of the whole audio (fine-tuned U2++). The archive's subtitles trail the speech.
# Not used yet: BV1rv4y1a7vK_p1 (单字表) and _p2 (词汇): items of 1–2 s with no stable lag, so cutting by
# subtitle time puts the wrong word in the clip.
SOURCES = [
    ("BV1ocq8Y5Eds", "yb", "语保·余姚 语法例句", [(0, None, "yubao", 0.0)]),
    # 慈溪话 (close to 余姚话; train only, the test split stays 余姚 WeChat voice)
    ("BV1DnqaYuEYn", "cx", "语保·慈溪 语法例句", [(0, None, "yubao", 0.0)]),
    # 慈溪（中部）方言记音 part 1: slides of 慈溪话 prose that the author reads, loosely (adds words): only
    # sentences whose recognized speech matches closely are kept. Parts 2–12 are phonetics (Mandarin text and
    # lists of example characters), not used.
    ("BV1zA411Y7Sn_p01", "cj", "慈溪方言记音 讲解", [(0, None, "slides", 0.0)]),
]
# Also not used: BV1rv4y1a7vK_p3. 0–127 s is speech (3 sentences, 乌鸦喝水, 为人民服务 retold in 余姚话) but its
# Mandarin subtitles don't line up with the utterances; from 127 s on it is 余姚滩簧 — singing, not speech.
HAN = re.compile(r"[㐀-鿿]")
JUNK = re.compile(r"[぀-ヿ]|发言人")  # kana = OCR noise; speaker captions


def range_at(ranges, t):
    for a, b, k, lag in ranges:
        if t >= a and (b is None or t < b):
            return k, lag
    return ranges[-1][2:]


def clean(s: str) -> str:
    return "".join(c for c in s if HAN.match(c) or c in "（）()~～")


def group_subtitles(rows, ranges):
    """Merge flickering OCR of one subtitle: consecutive, similar text, small gap. Text = longest-shown variant."""
    groups = []
    for r in rows:
        text = " ".join(r["lines"])
        if JUNK.search(text):
            continue
        t = clean(text)
        if not t:
            continue
        g = groups[-1] if groups else None
        if g and r["start"] - g["end"] <= 0.6 and difflib.SequenceMatcher(None, t, g["best"]).ratio() >= 0.5:
            g["end"] = r["end"]
            g["votes"][t] += r["end"] - r["start"]
            g["best"] = g["votes"].most_common(1)[0][0]
        else:
            groups.append({"start": r["start"], "end": r["end"], "votes": Counter({t: r["end"] - r["start"]}),
                           "best": t, "raw": text})
    out = []
    for g in groups:
        shown = g["end"] - g["start"]
        if shown < 0.4 or len(g["best"]) < 1:
            continue
        kind, lag = range_at(ranges, g["start"])
        out.append({"start": g["start"], "end": g["end"], "text": g["best"], "kind": kind, "lag": lag})
    return out


def group_yubao(rows):
    """语保 panel: one item per Mandarin sentence; keep the OCR with the most text (it has every line)."""
    groups = []
    for r in rows:
        head = clean(r["lines"][0] if r["lines"] else "")[:8]  # first Mandarin line
        if not head:
            continue
        g = groups[-1] if groups else None
        # strict: different sentences often share a few characters (墙上贴着… / 床上躺着…)
        if g and difflib.SequenceMatcher(None, head, g["head"]).ratio() >= 0.85:
            g["end"] = r["end"]
            if len("".join(r["lines"])) > len(g["raw"]):
                g["raw"] = "\n".join(r["lines"])
        else:
            groups.append({"start": r["start"], "end": r["end"], "head": head, "raw": "\n".join(r["lines"])})
    return [{"start": g["start"], "end": g["end"], "text": g["raw"], "kind": "yubao", "lag": 0.0} for g in groups
            if g["end"] - g["start"] >= 2]


# 慈溪 spellings outside the model vocabulary → the 说法表 / in-vocabulary character for the same word
CHAR_MAP = {"嗰": "葛"}


def normalize_spelling(text: str, glossary: list[dict]) -> str:
    for e in glossary:
        for a in e.get("avoid", []):
            text = text.replace(a, e["wu"])
    return "".join(CHAR_MAP.get(c, c) for c in text if c not in "=＝“”\"")


def slide_sentences(rows):
    """Slides that stay up ≥ 5 s, each split into sentences (with the slide's time window)."""
    out = []
    for r in rows:
        if r["end"] - r["start"] < 5:
            continue
        text = "".join(l for l in r["lines"] if not l.startswith(("本文参考", "参考")))
        text = re.sub(r"[•·]", "", text)
        for sent in re.split(r"[。！？；!?;]", text):
            if len(clean(sent)) >= 6:
                out.append({"start": r["start"], "end": r["end"], "text": sent.strip(), "kind": "slides", "lag": 0.0})
    return out


def timeline(stem: str, pcm: np.ndarray) -> list[tuple[float, str]]:
    """(seconds, character) for every character the fine-tuned CTC model (wu-ft) emits; cached per video."""
    path = os.path.join(DIR, "ocr", stem + ".ctc.json")
    if os.path.exists(path):
        return [tuple(x) for x in json.load(open(path, encoding="utf-8"))]
    import sherpa_onnx
    rec = sherpa_onnx.OfflineRecognizer.from_wenet_ctc(model=os.path.join(FT_DIR, "model.int8.onnx"),
                                                       tokens=os.path.join(FT_DIR, "tokens.txt"), num_threads=4)
    out, step = [], 20 * RATE
    for a in range(0, len(pcm), step):
        st = rec.create_stream()
        st.accept_waveform(RATE, (pcm[a:a + step] / 32768.0).tolist())
        rec.decode_stream(st)
        out += [(round(a / RATE + t, 2), c) for c, t in zip(st.result.tokens, st.result.timestamps)]
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return out


def dialect_line(raw: str) -> str:
    """语保 OCR: the dialect characters come after the IPA's closing bracket."""
    lines = raw.split("\n")
    last = max([i for i, l in enumerate(lines) if any(b in l for b in "]］」")] or [-1])
    return clean("".join(lines[last + 1:]))


def locate(tl, text: str, lo: float, hi: float):
    """The run of timeline characters in [lo, hi] that best covers `text` (bag of characters): returns
    (start s, end s, covered fraction) or None."""
    pts = [(t, c) for t, c in tl if lo <= t <= hi]
    want = Counter(text)
    best = None
    for i in range(len(pts)):
        have = Counter()
        for j in range(i, len(pts)):
            have[pts[j][1]] += 1
            hit = sum(min(n, have[c]) for c, n in want.items())
            score = hit - 0.5 * (j - i + 1 - hit)
            if best is None or score > best[0]:
                best = (score, pts[i][0], pts[j][0], hit / max(1, len(text)))
    if not best:
        return None
    # the model misrecognizes some characters, which the score counts against the run: extend it over
    # speech that continues without a pause (< 0.9 s between characters)
    times = [t for t, _ in tl]
    i, j = times.index(best[1]), len(times) - 1 - times[::-1].index(best[2])
    while i > 0 and times[i] - times[i - 1] < 0.9:
        i -= 1
    while j + 1 < len(times) and times[j + 1] - times[j] < 0.9:
        j += 1
    return times[i], times[j], best[3]


def build():
    os.makedirs(CLIPS, exist_ok=True)
    items = []
    conv = json.load(open(CONVERTED, encoding="utf-8")) if os.path.exists(CONVERTED) else {}
    proposals = {v["text"]: v["wu"] for v in conv.values() if v.get("text") and v.get("wu")}
    for stem, prefix, sender, ranges in SOURCES:
        rows = [json.loads(l) for l in open(os.path.join(DIR, "ocr", stem + ".jsonl"), encoding="utf-8")]
        kind0 = ranges[0][2]
        groups = (group_yubao(rows) if kind0 == "yubao" else slide_sentences(rows) if kind0 == "slides"
                  else group_subtitles(rows, ranges))
        pcm = read_pcm(to_wav(os.path.join(DIR, stem + ".m4a"), os.path.join(DIR, "wav", stem + ".wav")))
        tl = timeline(stem, pcm) if kind0 in ("yubao", "slides") else None
        cursor = 0.0  # slides: sentences are read in order
        kept = moved = dropped = 0
        for n, g in enumerate(groups, 1):
            # subtitles can appear a little before or vanish a little after the speech
            pad = (0, 0) if g["kind"] == "yubao" else (0.3, 0.5)
            a = max(0, int((g["start"] + g["lag"] - pad[0]) * RATE))
            b = min(len(pcm), int((g["end"] + g["lag"] + pad[1]) * RATE))
            if tl and g["kind"] == "slides":
                text = clean(g["text"])
                hit = locate(tl, text, max(g["start"] - 2, cursor), g["end"] + 2)
                if not hit:
                    dropped += 1
                    continue
                heard = [c for t, c in tl if hit[0] <= t <= hit[1]]
                have = Counter(heard)
                match = sum(min(n, have[c]) for c, n in Counter(text).items())
                # read nearly as written: most of the sentence heard, and little else said
                if hit[2] < 0.6 or match / max(1, len(heard)) < 0.6:
                    dropped += 1
                    continue
                cursor = hit[1]
                a, b = max(0, int((hit[0] - 0.35) * RATE)), min(len(pcm), int((hit[1] + 0.5) * RATE))
                moved += 1
            elif tl:
                # 语保: sometimes the recording plays an item while the next one is already on screen, or not at
                # all. Find where the item's dialect text is spoken (the cleaned-up proposal if there is one,
                # else the OCR'd dialect line); move the clip only if that covers the text much better than its
                # own panel time, and drop it if the text is spoken nowhere near.
                text = clean(proposals.get(g["text"], "")) or dialect_line(g["text"])
                if len(text) >= 3:
                    own = locate(tl, text, g["start"], g["end"])
                    near = locate(tl, text, g["start"] - 15, g["end"] + 15)
                    own_cov, near_cov = (own[2] if own else 0), (near[2] if near else 0)
                    if max(own_cov, near_cov) < 0.15:
                        dropped += 1
                        continue
                    if own_cov < 0.3 and near_cov - own_cov >= 0.2:
                        a, b = max(0, int((near[0] - 0.4) * RATE)), min(len(pcm), int((near[1] + 0.6) * RATE))
                        moved += 1
            span = speech_span(pcm[a:b])
            if not span:
                continue
            x = pcm[a + span[0]:a + span[1]]
            if len(x) < 0.3 * RATE or len(x) > 30 * RATE:
                continue
            key = f"{prefix}_{n:03d}.wav"
            write_wav(os.path.join(CLIPS, key), x)
            items.append({"key": key, "wav": os.path.join(CLIPS, key), "seconds": round(len(x) / RATE, 1),
                          "source": "video", "sender": sender, "kind": g["kind"], "text": g["text"],
                          "at": round((a + span[0]) / RATE, 2), "sub": [g["start"], g["end"]]})
            kept += 1
        print(f"{stem}: {len(rows)} OCR rows → {len(groups)} items → {kept} clips"
              + (f" ({moved} placed by their text, {dropped} dropped: not spoken, or not as written)" if tl else ""))
    with open(ITEMS, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} clips, {sum(i['seconds'] for i in items) / 60:.1f} min → {ITEMS}")


# ---------------------------------------------------------------- LLM proposals

RULES = """- wu：发音人实际说的余姚话，逐字照读音写，不要翻译成普通话。
  · 说法表里有的词一律用说法表的写法（看“别写成”，那些写法都要换掉）；说法表没有的，用读音最接近的常用汉字。
  · 重复、语气词照说的写，不要删也不要加。不要标点以外的符号（去掉“=”“□”之类）。
- mandarin：普通话意思，自然通顺，带标点。
- notes：拿不准的地方，一两句话，中文；没有就留空。"""

PROMPTS = {
    "yubao": """下面是“中国语言资源保护工程”{place}点一条语法例句的屏幕文字识别结果，可能有识别错误。依次是：普通话例句、
国际音标（方括号里）、方言字写法。录音里发音人读的是方言字写法那一句；有两种说法时（用“。”或“/”隔开）两句都读了，按顺序都写上。
“=”是语保标的同音替代字记号；“依”多半是“侬”识别错了；“日：”“目：”之类是识别出的杂字。请用音标核对、改正识别错误。

识别结果：
{text}

请输出：
""" + RULES,
    "wu": """下面是余姚话视频的一句字幕（屏幕文字识别，可能有错字），字幕用的写法未必和说法表一致。
另附本地语音识别模型对这段录音的草稿（按发音出字，错字很多，只用来核对字幕的读音和有没有错开）。

字幕：{text}
识别草稿：{drafts}

请输出：
""" + RULES + "\n  · 以字幕为准，只改明显的识别错字；草稿和字幕完全对不上时在 notes 里说明（可能错开了）。",
    "prompt": """这是《浙江方言语音档案·余姚话》的一段录音：屏幕上给出普通话提示，发音人用余姚话说出对应的说法
（词汇条目往往只说一个词，句子条目说一整句）。

普通话提示：{text}
本地语音识别模型对这段录音的草稿（按发音出字，错字很多）：{drafts}

请根据提示的意思和草稿的读音，推断发音人实际说的余姚话，输出：
""" + RULES + "\n  · 草稿和提示对不上时，在 notes 里说明。",
}


def propose(it: dict, glossary: str, drafts: dict, model: str = MODEL) -> dict:
    """GPT is text only: clips without on-screen dialect text use the local ASR drafts (whisper-wu / wu-local /
    wu-ft, filled in by the labeling server) as the evidence of what was said. A Gemini model also gets the clip
    (except 语保 items, whose dialect characters and IPA are on screen)."""
    from labeling.llm import llm_json  # lazy: keeps group_subtitles() importable without httpx
    d = drafts.get(it["key"], {})
    heard = "；".join(f"{n}：{t}" for n, t in d.items() if t) or "（无）"
    place = "慈溪" if "慈溪" in it["sender"] else "余姚"
    prompt = PROMPTS[it["kind"]].format(text=it["text"], drafts=heard, place=place)
    if place != "余姚":
        prompt += f"\n（这是{place}话，和余姚话很接近：和余姚话同一个词的用说法表写法，{place}特有的说法照原样写。）"
    prompt += "\n\n说法表（写法 = 普通话意思；别写成：…）：\n" + glossary
    wav = it["wav"] if model.startswith("gemini") and it["kind"] != "yubao" else None
    if wav:
        prompt += "\n\n另附这段录音：以录音为准，字幕、提示、草稿和录音不一样的地方照录音写，并在 notes 里说明。"
    return llm_json(model, prompt, ("wu", "mandarin", "notes"), wav) | {"model": model, "audio": bool(wav)}


def convert(workers: int, only: list[str] | None = None, limit: int | None = None, model: str = MODEL,
            redo: list[str] | None = None):
    gl = glossary_prompt()
    items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]
    done = json.load(open(CONVERTED, encoding="utf-8")) if os.path.exists(CONVERTED) else {}
    dr = asr_drafts()
    by_text = {v["text"]: v for v in done.values() if v.get("text")}
    for it in items:  # a rebuild can renumber clips: proposals follow their text, not the key
        if done.get(it["key"], {}).get("text") != it["text"]:
            done.pop(it["key"], None)
            if it["text"] in by_text:
                done[it["key"]] = by_text[it["text"]]
    for k in redo or []:
        done.pop(k, None)
    todo = [it for it in items if it["kind"] != "slides" and it["key"] not in done and (not only or it["key"].startswith(tuple(only)))][:limit]
    print(f"{len(todo)} to convert ({len(items)} items)")

    def one(it):
        try:
            return it["key"], propose(it, gl, dr, model) | {"text": it["text"]}
        except Exception as e:
            return it["key"], {"error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(workers) as ex:
        for n, (key, r) in enumerate(ex.map(one, todo), 1):
            if "error" in r:
                print(f"{key} ERROR {r['error'][:200]}", flush=True)
                continue
            done[key] = r
            print(f"[{n}/{len(todo)}] {key} {r['wu']}", flush=True)
            if n % 10 == 0 or n == len(todo):
                json.dump(done, open(CONVERTED + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                os.replace(CONVERTED + ".tmp", CONVERTED)
    json.dump(done, open(CONVERTED + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(CONVERTED + ".tmp", CONVERTED)


def publish():
    gloss = glossary()
    items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]
    conv = json.load(open(CONVERTED, encoding="utf-8")) if os.path.exists(CONVERTED) else {}
    out = []
    for it in items:
        c = conv.get(it["key"], {})
        prefill = normalize_spelling(it["text"], gloss) if it["kind"] == "slides" else c.get("wu")
        if not prefill:
            continue
        prompt = it["text"] if it["kind"] in ("prompt", "literal") else c.get("mandarin", "")
        out.append({"key": it["key"], "wav": it["wav"], "seconds": it["seconds"], "source": "video",
                    "sender": it["sender"], "prefill": prefill,
                    # 余姚 语保 sentences: the same fixed ~20% hash split as WeChat voices, a second test set
                    # (read speech by one speaker). 慈溪 and the course are train only.
                    "split": split_of(it["key"]) if it["key"].startswith("yb_") else "train",
                    "prompt": prompt})
    # the page lists newest first (reverses the list); write in reverse so a video reads top to bottom
    with open(EXTRA + ".tmp", "w", encoding="utf-8") as f:
        for e in reversed(out):
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(EXTRA + ".tmp", EXTRA)
    print(f"{len(out)} clips, {sum(e['seconds'] for e in out) / 60:.1f} min → {EXTRA}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    c = sub.add_parser("convert")
    c.add_argument("--workers", type=int, default=12)
    c.add_argument("--only", nargs="*", help="key prefixes, e.g. yb_ da2_00")
    c.add_argument("--limit", type=int)
    c.add_argument("--model", default=MODEL, help="gemini-4-hydrogen also sends the audio (paid; ask first)")
    c.add_argument("--redo", nargs="*", help="keys to convert again even if cached")
    sub.add_parser("publish")
    a = ap.parse_args()
    {"build": build, "convert": lambda: convert(a.workers, a.only, a.limit, a.model, a.redo), "publish": publish}[a.cmd]()


if __name__ == "__main__":
    main()
