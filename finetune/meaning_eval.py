#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "wenet @ https://github.com/wenet-e2e/wenet/archive/be6ad97fdb512589df113042f7ff3a29d392fb38.tar.gz",
#   "torch==2.5.1", "torchaudio==2.5.1", "numpy<2", "pyyaml", "openai-whisper", "librosa",
#   "httpx",  # stt/refine.py
# ]
# ///
"""Judge models by what the bot does with their output: transcript → deepseek-flash (stt/refine.py, the same
prompt as the combo provider) → Mandarin, compared with the labeled Mandarin meaning. CER counts every
particle; a misheard 啦/嘛 or 葛/个 doesn't change what the reply should be, a dropped 勿 does.

  uv run finetune/meaning_eval.py decode NAME KIND CKPT     test-set transcripts → data/labels/eval/hyp-NAME.json
                                                             (KIND: whisper | u2pp: WeNet checkpoint;
                                                              deployed: openai-whisper .pt, fp16 beam 5 as the bot)
  uv run finetune/meaning_eval.py judge NAME[+NAME2] ...     refine + judge each, print a table
        [--llm gpt-6.2-sol]                                  refine with an OpenAI model instead of deepseek-flash
        [--runs 5]                                           refine each input 5 times, report mean and range

NAME "label" is the human 余姚话 label itself (the ceiling: what the LLM makes of a perfect transcript).
"A+B" gives the LLM both transcripts, as the combo provider can. Refined Mandarin and verdicts are cached in
data/labels/eval/ (keyed by the transcripts and the refine prompt), so re-running only calls DeepSeek for new inputs.
Judge: deepseek-flash, thinking off, scores 2 = same meaning, 1 = gist right but a detail wrong/missing,
0 = wrong or opposite. Particles, tone, word choice and punctuation don't count.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stt.cer import edits  # noqa: E402
from stt.paths import LABELS as LABEL_DIR  # noqa: E402

EVAL = os.path.join(LABEL_DIR, "eval")
LABELS = os.path.join(LABEL_DIR, "labels.jsonl")
TEST = os.path.join(LABEL_DIR, "wenet", "test.jsonl")


def test_items() -> list[dict]:
    meaning = {}
    for line in open(LABELS, encoding="utf-8"):
        r = json.loads(line)
        meaning[r["key"].removesuffix(".wav")] = r["meaning"]  # latest line wins
    items = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    for it in items:
        it["meaning"] = meaning.get(it["key"], "")
    return items


def group(key: str) -> str:
    return "语保" if key.startswith("yb_") else "wechat"


# ---------------------------------------------------------------- decode

def decode(name: str, kind: str, ckpt: str):
    import torch
    if kind == "deployed":  # an openai-whisper .pt, decoded exactly as the whisper-ft provider does
        from stt import local_whisper_wu
        from stt.audio import Audio
        out = {it["key"]: local_whisper_wu.transcribe(Audio(it["wav"]), ckpt=ckpt) for it in test_items()}
        out = {k: re.sub(r"[^㐀-鿿]", "", v) for k, v in out.items()}
    elif kind == "u2pp":
        import u2pp
        model = u2pp.load_model(ckpt)
        model.eval()
        id2ch = {i: c for c, i in u2pp.vocab().items()}
        out = {}
        with torch.no_grad():
            for it in u2pp.load_split("test"):
                b = u2pp.batchify([it], train=False)
                res = model.decode(["attention_rescoring"], b["feats"], b["feats_lengths"], beam_size=10,
                                   ctc_weight=0.3, reverse_weight=0.3)
                out[it["key"]] = "".join(id2ch.get(t, "") for t in res["attention_rescoring"][0].tokens)
    else:
        import whisper_ft as w
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model = w.load_model(ckpt).to(dev)
        model.eval()
        han = re.compile(r"[^㐀-鿿]")
        out = {}
        with torch.no_grad():
            for it in w.load_split("test"):
                b = w.batchify([it], train=False)
                try:
                    res = model.decode(["attention"], b["feats"].to(dev), b["feats_lengths"].to(dev), beam_size=1,
                                       infos={"tasks": ["transcribe"], "langs": ["zh"]})
                    ids = [t for t in res["attention"][0].tokens if t < w.tokenizer().eot]
                    out[it["key"]] = han.sub("", w.tokenizer().decode(ids))
                except AssertionError:  # decoder looped past 448 positions
                    out[it["key"]] = ""
    os.makedirs(EVAL, exist_ok=True)
    path = os.path.join(EVAL, f"hyp-{name}.json")
    json.dump({"ckpt": ckpt, "kind": kind, "hyp": out}, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(out)} transcripts → {path}")


# ---------------------------------------------------------------- judge

JUDGE = """你在评估一个余姚话语音转写系统。下面是同一条语音的两句普通话：
【标准意思】是人工标注的正确意思；【系统输出】是系统识别后翻译成的普通话。
判断【系统输出】是否传达了【标准意思】：
2 = 意思相同（用词、语气词、标点、语序不同都没关系）
1 = 大意对，但有次要信息错了或缺了
0 = 意思错了、相反（比如漏了否定），或者关键信息错了
只输出 JSON：{"score": 0/1/2, "why": "一句话理由"}

【标准意思】%s
【系统输出】%s"""


def _cache(name: str) -> dict:
    p = os.path.join(EVAL, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def _save(name: str, d: dict):
    os.makedirs(EVAL, exist_ok=True)
    json.dump(d, open(os.path.join(EVAL, name), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def judge_one(ref: str, out: str) -> dict:
    from stt import refine
    import httpx
    from stt import keys
    body = {"model": refine.MODEL, "thinking": {"type": "disabled"}, "temperature": 0,
            "messages": [{"role": "user", "content": JUDGE % (ref, out)}]}
    r = httpx.post(refine.URL, headers={"Authorization": f"Bearer {keys.get('deepseek_api_key')}"},
                   json=body, timeout=60)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))


def refine_openai(model: str, cands: list[str]) -> str:
    """Same system prompt and user message as stt/refine.py, answered by an OpenAI model (Responses API)."""
    import httpx
    from stt import keys, refine
    user = "\n".join(f"识别结果{i + 1}：{c}" for i, c in enumerate(cands))
    body = {"model": model, "instructions": refine.system_prompt(), "input": user}
    r = httpx.post("https://api.openai.com/v1/responses", json=body, timeout=120,
                   headers={"Authorization": f"Bearer {keys.get('openai_api_key')}"})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {' '.join(r.text.split())[:300]}")
    return "".join(c.get("text", "") for item in r.json().get("output", []) if item.get("type") == "message"
                   for c in item.get("content", []) if c.get("type") == "output_text").strip()


def judge(names: list[str], llm: str = "deepseek", runs: int = 1):
    """runs > 1: refine every input that many times (the LLM isn't deterministic) and average the scores."""
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    from stt import refine
    items = test_items()
    ptag = hashlib.md5(refine.system_prompt().encode()).hexdigest()[:8]  # new 说法表/notes → new refinements
    rname = f"refined-{ptag}.json" if llm == "deepseek" else f"refined-{llm}-{ptag}.json"
    refined, verdicts = _cache(rname), _cache("verdicts.json")
    hyps = {"label": {it["key"]: it["txt"] for it in items}}
    for n in {p for name in names for p in name.split("+")} - {"label"}:
        hyps[n] = json.load(open(os.path.join(EVAL, f"hyp-{n}.json"), encoding="utf-8"))["hyp"]

    def job(name, key, r):  # cache key: the transcripts, plus the run number after the first run
        return json.dumps([hyps[p].get(key, "") for p in name.split("+")], ensure_ascii=False) + (f"#{r}" if r else "")

    def do_refine(j):
        cands = [c for c in json.loads(j.split("#")[0]) if c] or ["（无识别结果）"]
        try:
            return j, refine.refine(cands) if llm == "deepseek" else refine_openai(llm, cands)
        except Exception as e:
            return j, f"✗ {type(e).__name__}: {e}"
    todo = sorted({job(n, it["key"], r) for n in names for it in items for r in range(runs)} - set(refined))
    with ThreadPoolExecutor(8) as pool:
        for j, out in pool.map(do_refine, todo):
            refined[j] = out
    _save(rname, refined)

    pairs = {(it["meaning"], refined[job(n, it["key"], r)]) for n in names for it in items for r in range(runs)}
    todo = [p for p in pairs if json.dumps(p, ensure_ascii=False) not in verdicts]
    with ThreadPoolExecutor(8) as pool:
        for p, v in zip(todo, pool.map(lambda p: judge_one(*p), todo)):
            verdicts[json.dumps(p, ensure_ascii=False)] = v
    _save("verdicts.json", verdicts)

    import u2pp
    rows, detail = [], []
    for n in names:
        cells = []
        for gname in ("wechat", "语保"):
            its = [it for it in items if group(it["key"]) == gname]
            if not its:
                continue
            per_run = []
            for r in range(runs):
                s = []
                for it in its:
                    out = refined[job(n, it["key"], r)]
                    v = verdicts[json.dumps((it["meaning"], out), ensure_ascii=False)]
                    s.append(v["score"])
                    if r == 0:
                        detail.append((it["key"], n, " / ".join(json.loads(job(n, it["key"], 0))), out, v))
                per_run.append(sum(s) / (2 * len(s)))
            e = sum(edits(hyps[n].get(it["key"], ""), it["txt"]) for it in its) if "+" not in n else None
            cer = f"CER {e / sum(len(it['txt']) for it in its):5.1%}" if e is not None else "CER   —  "
            spread = f" ({min(per_run):.0%}–{max(per_run):.0%})" if runs > 1 else ""
            cells.append(f"{gname} {len(its):2d}: score {sum(per_run) / runs:5.1%}{spread:12s} {cer}")
        rows.append(f"{n:22s} " + "   |   ".join(cells))
    if runs == 1:
        for key, n, hyp, out, v in sorted(detail):
            print(f"{key:22s} {n:18s} {'✗△✓'[v['score']]}  {hyp[:40]}  →  {out[:40]}   ({v['why'][:40]})")
        print()
    print(f"refine {llm} · prompt {ptag} · {runs} run(s), score = mean of ✓=1 △=½ ✗=0 (range over runs)")
    print("\n".join(rows))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode")
    d.add_argument("name")
    d.add_argument("kind", choices=["whisper", "u2pp", "deployed"])
    d.add_argument("ckpt")
    j = sub.add_parser("judge")
    j.add_argument("names", nargs="+")
    j.add_argument("--llm", default="deepseek", help='"deepseek" (the bot) or an OpenAI model, e.g. gpt-6.2-sol')
    j.add_argument("--runs", type=int, default=1, help="refine each input N times and average")
    a = p.parse_args()
    if a.cmd == "decode":
        decode(a.name, a.kind, a.ckpt)
    else:
        judge(a.names, a.llm, a.runs)


if __name__ == "__main__":
    main()
