#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pypinyin"]
# ///
"""Reading sheets: 余姚话 sentences for people to read aloud and send back as WeChat voice messages.

  uv run stt/read_sheets.py check     problems in the sentences + 说法表 coverage
  uv run stt/read_sheets.py sheets    data/read/sheets/sheet-NN.txt (+ .json with the sentence ids)

Sentences: data/read/sentences.tsv, one per line: id <TAB> 余姚话 (说法表 spellings) <TAB> 普通话.
Reading hints: data/read/readings.tsv, `key <TAB> reading`, where key is a character or a word (words win,
longest first, so polyphones like 困晏觉 can be fixed in one place). A character with no entry falls back
to its Mandarin pinyin; `check` lists those so they can be given a 余姚 reading. The hint is pinyin-style
(no tones), only meant to help the reader recognise the word, not a phonetic transcription.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "read")
SENTENCES = os.path.join(DIR, "sentences.tsv")
READINGS = os.path.join(DIR, "readings.tsv")
SHEETS = os.path.join(DIR, "sheets")
GLOSSARY = os.path.join(ROOT, "data", "labels", "glossary.json")
TOKENS = os.path.join(ROOT, "models", "wu-u2pp-int8", "tokens.txt")
PUNCT = {"，": ",", "？": "?", "！": "!", "。": "", "、": ",", "：": ":", "；": ";"}
HAN = re.compile(r"[㐀-鿿]")

HEADER = """余姚话朗读单 第 {n} 张（{k} 句）

请按顺序，一句发一条微信语音给我。
· 像平时讲话一样随口说，不用一个字一个字念。
· 觉得哪句不像余姚话，就按你自己的说法讲，再打字告诉我改了哪里。
· 读错了就重新发一条，不用删。
每句三行：余姚话 / 拼音提示（照普通话拼音读，只是帮你认字）/ 普通话意思。
"""


def sentences() -> list[dict]:
    out = []
    for n, line in enumerate(open(SENTENCES, encoding="utf-8"), 1):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            sys.exit(f"{SENTENCES}:{n}: need id<TAB>余姚话<TAB>普通话: {line!r}")
        out.append({"id": parts[0], "wu": parts[1].strip(), "zh": parts[2].strip()})
    return out


def readings() -> dict[str, str]:
    r = {}
    if os.path.exists(READINGS):
        for line in open(READINGS, encoding="utf-8"):
            if line.strip() and not line.startswith("#"):
                k, v = line.rstrip("\n").split("\t")
                r[k] = v.strip()
    return r


def hint(wu: str, rd: dict[str, str], missing: Counter | None = None) -> str:
    from pypinyin import lazy_pinyin
    longest = max(map(len, rd), default=1)
    out, i = [], 0
    while i < len(wu):
        c = wu[i]
        if c in PUNCT:
            if out and PUNCT[c]:
                out[-1] += PUNCT[c]
            i += 1
            continue
        for n in range(min(longest, len(wu) - i), 0, -1):
            if wu[i:i + n] in rd:
                out.append(rd[wu[i:i + n]])
                i += n
                break
        else:
            if HAN.match(c):
                out.append(lazy_pinyin(c, v_to_u=True)[0])
                if missing is not None:
                    missing[c] += 1
            elif not c.isspace():
                out.append(c)
            i += 1
    return " ".join(out)


def check():
    ss, rd = sentences(), readings()
    vocab = {l.split()[0] for l in open(TOKENS, encoding="utf-8")}
    gl = json.load(open(GLOSSARY, encoding="utf-8"))
    problems = []
    ids = Counter(s["id"] for s in ss)
    problems += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    texts = Counter(s["wu"] for s in ss)
    problems += [f"duplicate sentence {t}" for t, n in texts.items() if n > 1]
    missing = Counter()
    for s in ss:
        bare = "".join(c for c in s["wu"] if c not in PUNCT)
        oov = {c for c in bare if c not in vocab}
        if oov:
            problems.append(f"{s['id']} not in model vocab: {''.join(sorted(oov))}  {s['wu']}")
        for e in gl:
            for a in e.get("avoid", []):
                if a in s["wu"]:
                    problems.append(f"{s['id']} uses {a} (说法表: {e['wu']})  {s['wu']}")
        if not 4 <= len(bare) <= 26:
            problems.append(f"{s['id']} {len(bare)} chars  {s['wu']}")
        hint(s["wu"], rd, missing)
    print(f"{len(ss)} sentences, {sum(len(s['wu']) for s in ss)} chars")
    print("\n".join(problems) or "no problems")
    if missing:
        print(f"\nno 余姚 reading yet (Mandarin pinyin used), {len(missing)} chars:")
        print(" ".join(f"{c}{n}" for c, n in missing.most_common()))
    used = Counter()
    for e in gl:
        used[e["wu"]] = sum(s["wu"].count(e["wu"]) for s in ss)
    unused = [w for w, n in used.items() if n == 0]
    print(f"\n说法表 coverage: {len(gl) - len(unused)}/{len(gl)} entries used; unused:")
    print(" ".join(unused))


def sheets(per: int):
    ss, rd = sentences(), readings()
    os.makedirs(SHEETS, exist_ok=True)
    for k in range(0, len(ss), per):
        part, n = ss[k:k + per], k // per + 1
        lines = [HEADER.format(n=n, k=len(part))]
        for j, s in enumerate(part, 1):
            lines.append(f"{j}. {s['wu']}\n{hint(s['wu'], rd)}\n（{s['zh']}）\n")
        path = os.path.join(SHEETS, f"sheet-{n:02d}")
        open(path + ".txt", "w", encoding="utf-8").write("\n".join(lines))
        json.dump([s["id"] for s in part], open(path + ".json", "w"), indent=0)
        print(path + ".txt", len(part))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    s = sub.add_parser("sheets")
    s.add_argument("--per", type=int, default=50)
    a = ap.parse_args()
    check() if a.cmd == "check" else sheets(a.per)


if __name__ == "__main__":
    main()
