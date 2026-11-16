"""Merge several ASR transcripts of one clip into Mandarin with an LLM that knows 余姚话.

Speech APIs either paraphrase Wu dialect into Mandarin (Volcengine) or spell its sounds with
homophone characters (侬来弄后戏啦 — DeepSeek/OpenAI). Given all of them, an LLM recovers the
meaning far more reliably than any single API. Tested 2026-11-14: deepseek-flash with thinking
disabled was right 12/12 at ~1.1 s; with thinking on it sometimes ran 20–30 s; gpt-5.4-mini and
gpt-5.5 tended to copy the Volcengine candidate's mistakes.
"""
import re

import httpx

from . import keys

URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"

SYSTEM = (
    "你负责把一条微信语音的识别结果整理成普通话。说话人讲普通话，或者讲余姚话"
    "（浙江宁波余姚的吴语方言，和上海话、宁波话很接近）。\n"
    "下面是几个识别系统对【同一句话】的转写，每个都可能有错：有的会把方言直接意译成普通话，"
    "有的会用同音或近音汉字记录吴语读音，字面看起来不通顺。\n"
    "规则：\n"
    "1. 这些结果说的是同一句话，综合它们的读音和意思，推断说话人实际说的那一句；绝不能把几个结果拼接起来。\n"
    "2. 如果是吴语，按吴语的读音和词汇翻译成自然的普通话。\n"
    "3. 如果结果本来就是通顺的普通话且彼此一致，原样输出，不要替换用词。\n"
    "只输出这一句普通话简体中文，不要解释。"
)


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
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
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
        if not META.search(text):
            return text
    raise RuntimeError(f"LLM answered with commentary instead of a sentence: {text[:80]}")
