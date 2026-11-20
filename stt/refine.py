"""Merge several ASR transcripts of one clip into Mandarin with an LLM that knows 余姚话.

Speech APIs either paraphrase Wu dialect into Mandarin (Volcengine) or spell its sounds with
homophone characters (侬来弄后戏啦 — DeepSeek/OpenAI). Given all of them, an LLM recovers the
meaning far more reliably than any single API. Tested 2026-11-14: deepseek-flash with thinking
disabled was right 12/12 at ~1.1 s; with thinking on it sometimes ran 20–30 s; gpt-5.4-mini and
gpt-5.5 tended to copy the Volcengine candidate's mistakes.
"""
import json
import os
import re

import httpx

from . import keys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yuyao_notes.md")  # grammar and particles
GLOSSARY = os.path.join(ROOT, "data", "labels", "glossary.json")  # 说法表 (labeling page), if present

URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"

SYSTEM = (
    "你负责把一条微信语音的识别结果整理成普通话。说话人讲普通话，或者讲余姚话"
    "（浙江宁波余姚的吴语方言，和上海话、宁波话很接近）。\n"
    "下面是识别系统对【同一句话】的转写（一个或几个），都可能有错：有的会把方言直接意译成普通话，"
    "有的会用同音或近音汉字记录吴语读音，字面看起来不通顺。\n"
    "规则：\n"
    "1. 有几个结果时，它们说的是同一句话，综合它们的读音和意思，推断说话人实际说的那一句；绝不能把几个结果拼接起来。\n"
    "2. 如果是吴语，按吴语的读音和词汇翻译成自然的普通话。\n"
    "3. 如果结果本来就是通顺的普通话（有几个时彼此一致），原样输出，不要替换用词。\n"
    "4. 理解余姚话时参考下面的要点和说法表。\n"
)
END = "\n\n只输出这一句普通话简体中文，不要解释。"
_system = (None, "")


def system_prompt() -> str:
    """SYSTEM + 余姚话 notes + 说法表, rebuilt when either file changes."""
    global _system
    stamp = tuple(os.path.getmtime(f) if os.path.exists(f) else 0 for f in (NOTES, GLOSSARY))
    if _system[0] != stamp:
        parts = [SYSTEM]
        if os.path.exists(NOTES):
            parts.append(open(NOTES, encoding="utf-8").read())
        if os.path.exists(GLOSSARY):
            gl = json.load(open(GLOSSARY, encoding="utf-8"))
            parts.append("## 说法表（余姚话写法 = 普通话意思）\n" + "\n".join(f"{e['wu']} = {e['mandarin']}" for e in gl))
        _system = (stamp, "\n\n".join(parts) + END)
    return _system[1]


def ready() -> bool:
    return bool(keys.get("deepseek_api_key"))


# On garbled long clips the LLM occasionally answers about the task instead of with a sentence
# ("这些识别结果都不太靠谱…请提供更准确的识别结果"). A prompt rule against it did not help (2026-11-15),
# so such replies are detected here, retried once, then reported as an error (combo falls back).
META = re.compile(r"识别结果|识别系统|请提供|不太靠谱|无法(可靠|确定|推断|还原|判断)|整理成普通话|抱歉")


def _ask(user: str) -> str:
    body = {
        "model": MODEL,
        "thinking": {"type": "disabled"},  # thinking adds 10–30 s on ambiguous input, no gain
        "messages": [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}],
    }
    r = httpx.post(URL, headers={"Authorization": f"Bearer {keys.get('deepseek_api_key')}"},
                   json=body, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {' '.join(r.text.split())[:300]}")
    return r.json()["choices"][0]["message"]["content"].strip()


def refine(candidates: list[str]) -> str:
    user = "\n".join(f"识别结果{i + 1}：{c}" for i, c in enumerate(candidates))
    for _ in range(2):
        text = _ask(user)
        # also reject near-empty answers (seen once with the long prompt: just "的")
        if not META.search(text) and len(re.findall(r"[\u3400-\u9fff]", text)) >= 2:
            return text
    raise RuntimeError(f"LLM answered with commentary instead of a sentence: {text[:80]}")
