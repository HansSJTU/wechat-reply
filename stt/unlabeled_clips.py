#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "sherpa-onnx", "httpx", "websockets>=13"]
# ///
"""Cut downloaded 余姚话 / 慈溪话 audio without transcripts into utterance clips for the labeling page.

  uv run stt/unlabeled_clips.py build [--max-per-file 60]   clips → data/bili/unlabeled/clips/, items.jsonl
  uv run stt/unlabeled_clips.py publish                      → data/labels/extra_clips_web.jsonl (page tab 网络)
  uv run stt/unlabeled_clips.py words                        word-list readings → data/labels/extra_clips_words.jsonl
                                                             (page tab 词组; see WORD_SOURCES)

Input: data/bili/unlabeled/{yuyao,cixi}/*.m4a and manifest.jsonl (one line per file: file, title, uploader,
dialect, type, …). Utterances are found by loudness (10 ms frames above -30 dB of the loud level and 3× the
noise floor), short pauses (< 0.6 s) are bridged, runs are joined up to ~12 s, longer ones are split at their
quietest point, and 0.2 s is kept on each side. Clips under 1 s are dropped. Music under speech defeats this;
such clips get marked 坏音频 / 不使用 on the page.
TV programs mix the host's 余姚话 with Mandarin interviews and music, so every clip is first recognized with
wu-ft (fast, CPU) and kept only if the draft reads as dialect: more 余姚话 marker words (侬 阿拉 葛 啷 哉 勿 呒
伐 渠…) than Mandarin ones (的 我们 这个 就是 没有…), and at least a few characters (music gives "嗯").
The labeling server then drafts the kept clips with all local models. All clips are train only.
"""
import argparse
import json
import os
import subprocess
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DIR = os.path.join(ROOT, "data", "bili", "unlabeled")
CLIPS = os.path.join(DIR, "clips")
ITEMS = os.path.join(DIR, "items.jsonl")
OUT = os.path.join(ROOT, "data", "labels", "extra_clips_web.jsonl")
RATE = 16000
HOP = RATE // 100  # 10 ms


def to_pcm(path: str) -> np.ndarray:
    wav = os.path.join(DIR, "wav", os.path.splitext(os.path.basename(path))[0] + ".wav")
    if not os.path.exists(wav):
        os.makedirs(os.path.dirname(wav), exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "1", "-ar", str(RATE), wav + ".tmp.wav"],
                       check=True)
        os.replace(wav + ".tmp.wav", wav)
    with wave.open(wav) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def utterances(x: np.ndarray, max_len: float = 12.0, min_len: float = 1.0, bridge: float = 0.6,
               pad: float = 0.2) -> list[tuple[int, int]]:
    """(start, end) sample ranges of speech; pauses shorter than `bridge` seconds are joined."""
    n = len(x) // HOP
    if n < 10:
        return []
    rms = np.sqrt((x[:n * HOP].astype(np.float64).reshape(n, HOP) ** 2).mean(1))
    thr = max(np.percentile(rms, 99) * 0.0316, 3 * np.percentile(rms, 10), 30.0)
    loud = rms > thr
    runs, i = [], 0
    while i < n:  # loud runs
        if loud[i]:
            j = i
            while j < n and loud[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    merged = []  # bridge short pauses, join up to max_len
    for a, b in runs:
        if merged and a - merged[-1][1] < bridge * 100 and b - merged[-1][0] <= max_len * 100:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []

    def split(a, b):
        if b - a <= max_len * 100 * 1.25:
            out.append((a, b))
            return
        lo, hi = a + (b - a) // 4, b - (b - a) // 4  # cut at the quietest 100 ms in the middle half
        win = np.convolve(rms[lo:hi], np.ones(10), "valid")
        cut = lo + int(np.argmin(win)) + 5
        split(a, cut)
        split(cut, b)

    for a, b in merged:
        split(a, b)
    p = int(pad * 100)
    return [(max(0, a - p) * HOP, min(n, b + p) * HOP) for a, b in out if (b - a) >= min_len * 100]


WU = ["侬", "阿拉", "㑚", "葛", "啷", "哉", "勿", "呒", "伐", "渠", "噶", "搿", "杂格", "啥人", "晓得", "交关", "老老",
      "来葛", "辣盖", "末", "介", "囥", "驮", "汏", "忖", "首西", "色格", "噎", "卯", "和水", "宁"]
MANDARIN = ["的", "我们", "他们", "你们", "这个", "那个", "就是", "没有", "什么", "觉得", "非常", "因为", "所以",
            "一些", "现在", "可以", "这样", "怎么", "知道", "吗", "呢", "很多"]


def dialect_score(text: str) -> tuple[int, int]:
    return sum(text.count(w) for w in WU), sum(text.count(w) for w in MANDARIN)


def is_dialect(text: str, seconds: float) -> bool:
    han = sum(1 for c in text if "\u3400" <= c <= "\u9fff")
    if han < max(3, seconds * 1.0):  # music / silence / a lone 嗯
        return False
    wu, md = dialect_score(text)
    return wu >= 2 and wu > md


def build(max_per_file: int):
    from types import SimpleNamespace
    from stt import local_wu
    rows = [json.loads(l) for l in open(os.path.join(DIR, "manifest.jsonl"), encoding="utf-8") if l.strip()]
    os.makedirs(CLIPS, exist_ok=True)
    items, total = [], 0.0
    for r in rows:
        path = r["file"] if os.path.isabs(r["file"]) else os.path.join(DIR, r["file"])
        if not os.path.exists(path):
            path = os.path.join(DIR, r.get("dialect_dir", ""), os.path.basename(r["file"]))
        if not os.path.exists(path):
            print("missing", r["file"])
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        prefix = "yy" if "余姚" in r.get("dialect", "") or "/yuyao/" in path else "cx"
        x = to_pcm(path)
        segs = utterances(x)
        kept_segs = []
        for k, (a, b) in enumerate(segs, 1):
            sec = (b - a) / RATE
            text = local_wu.transcribe(SimpleNamespace(pcm=x[a:b].tobytes()), local_wu.FT_DIR)  # only .pcm is read
            if is_dialect(text, sec):
                kept_segs.append((k, a, b))
        n_dialect = len(kept_segs)
        if max_per_file and len(kept_segs) > max_per_file:  # spread over the whole file
            idx = np.linspace(0, len(kept_segs) - 1, max_per_file).round().astype(int)
            kept_segs = [kept_segs[i] for i in sorted(set(idx))]
        for k, a, b in kept_segs:
            key = f"{prefix}_{stem}_{k:04d}.wav"
            dst = os.path.join(CLIPS, key)
            with wave.open(dst + ".tmp", "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(x[a:b].tobytes())
            os.replace(dst + ".tmp", dst)
            sec = (b - a) / RATE
            total += sec
            items.append({"key": key, "wav": dst, "seconds": round(sec, 1), "source": "web",
                          "sender": f"{r.get('dialect', '')} · {r.get('uploader', '')} · {r.get('title', '')[:30]}",
                          "split": "train", "at": round(a / RATE, 2), "bv": r.get("bv")})
        print(f"{stem}: {len(x) / RATE / 60:5.1f} min → {len(segs)} utterances, {n_dialect} dialect → {len(kept_segs)} kept"
              f"  {r.get('title', '')[:30]}", flush=True)
    with open(ITEMS, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} clips, {total / 60:.1f} min → {ITEMS}")


# Word / character list readings: one word per utterance, no dialect filter (short, often a single word),
# kept apart (tab 词组) so their share of the training data can be controlled.
# (audio, key prefix, sender, OCR of the on-screen prompts or None)
WORD_SOURCES = [
    ("data/bili/BV1rv4y1a7vK_p1.m4a", "w_yyzi", "余姚 · 方言档案 · 单字/词 朗读", "data/bili/ocr/BV1rv4y1a7vK_p1.jsonl"),
    ("data/bili/BV1rv4y1a7vK_p2.m4a", "w_yyci", "余姚 · 方言档案 · 词汇（看普通话说方言）", "data/bili/ocr/BV1rv4y1a7vK_p2.jsonl"),
    ("data/bili/unlabeled/cixi/BV1Ps4y1f7Wk_p01.m4a", "w_cxda", "慈溪 · 方言视频档案", None),
    ("data/bili/unlabeled/cixi/BV1Tr4y1P77K_p01.m4a", "w_cxdb", "慈溪 · 慈城话 vs 浒山话 对比", None),
]
OUT_WORDS = os.path.join(ROOT, "data", "labels", "extra_clips_words.jsonl")


def ocr_hints(path: str) -> list[tuple[float, float, str]]:
    """On-screen prompts (deduplicated OCR) as (start, end, text)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bili_clips import group_subtitles
    rows = [json.loads(l) for l in open(os.path.join(ROOT, path), encoding="utf-8")]
    return [(g["start"], g["end"], g["text"]) for g in group_subtitles(rows, [(0, None, "prompt", 0.0)])]


def words():
    os.makedirs(CLIPS, exist_ok=True)
    out, total = [], 0.0
    for audio, prefix, sender, ocr in WORD_SOURCES:
        x = to_pcm(os.path.join(ROOT, audio))
        segs = utterances(x, max_len=4.0, min_len=0.3, bridge=0.25, pad=0.15)
        hints = ocr_hints(ocr) if ocr else []
        for k, (a, b) in enumerate(segs, 1):
            key = f"{prefix}_{k:04d}.wav"
            dst = os.path.join(CLIPS, key)
            with wave.open(dst + ".tmp", "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(x[a:b].tobytes())
            os.replace(dst + ".tmp", dst)
            t0, t1 = a / RATE, b / RATE
            # the prompt shown around then (subtitles are only roughly in sync: a hint, not the label)
            near = [t for s0, s1, t in hints if s0 <= t1 + 2 and s1 >= t0 - 2]
            sec = (b - a) / RATE
            total += sec
            out.append({"key": key, "wav": dst, "seconds": round(sec, 1), "source": "words", "sender": sender,
                        "split": "train", "at": round(t0, 2),
                        "prompt": ("画面提示（时间只大致对得上）：" + " / ".join(near[:3])) if near else ""})
        print(f"{audio}: {len(x) / RATE / 60:.1f} min → {len(segs)} clips", flush=True)
    with open(OUT_WORDS + ".tmp", "w", encoding="utf-8") as f:
        for it in reversed(out):
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    os.replace(OUT_WORDS + ".tmp", OUT_WORDS)
    print(f"{len(out)} clips, {total / 60:.1f} min → {OUT_WORDS}")


def publish():
    items = [json.loads(l) for l in open(ITEMS, encoding="utf-8")]
    with open(OUT + ".tmp", "w", encoding="utf-8") as f:
        for it in reversed(items):  # the page lists newest first
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    os.replace(OUT + ".tmp", OUT)
    print(f"{len(items)} clips → {OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--max-per-file", type=int, default=40, help="0 = all; keeps the page and drafting manageable")
    sub.add_parser("publish")
    sub.add_parser("words")
    a = ap.parse_args()
    {"build": lambda: build(a.max_per_file), "publish": publish, "words": words}[a.cmd]()


if __name__ == "__main__":
    main()
