"""Placeholder reply model: an OpenAI chat bot (Responses API) that looks at images and can search the web."""
import base64
import json
import datetime
import mimetypes
import re

import httpx

from stt import keys

from .log import log

URL = "https://api.openai.com/v1/responses"
IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
WEEKDAYS = "一二三四五六日"
# web search answers carry markdown citations; WeChat shows them raw
LINK = re.compile(r"\s*\(\[([^\]]*)\]\([^)]*\)\)|\[([^\]]*)\]\(https?://[^)]*\)")


def _image_part(path):
    mime = mimetypes.guess_type(path)[0]
    if mime not in IMAGE_TYPES:
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"type": "input_image", "image_url": f"data:{mime};base64,{data}"}


def build_input(history, max_images):
    """history: [{role, text, image_path}] oldest first. Only the newest `max_images` are attached."""
    with_image = [i for i, h in enumerate(history) if h.get("image_path")][-max_images:] if max_images else []
    out = []
    for i, h in enumerate(history):
        if h["role"] == "assistant":
            out.append({"role": "assistant", "content": [{"type": "output_text", "text": h["text"]}]})
            continue
        content = [{"type": "input_text", "text": h["text"]}]
        part = _image_part(h["image_path"]) if i in with_image else None
        if part:
            content.append(part)
        out.append({"role": "user", "content": content})
    return out


def _clean(text):
    text = LINK.sub(lambda m: m.group(2) or "", text)  # drop "([site](url))", keep the text of "[text](url)"
    return text.replace("**", "").strip()


SILENT = "[不回]"  # an unprompted group reply may still back out with exactly this


def _post(cfg, body):
    r = httpx.post(URL, headers={"Authorization": f"Bearer {keys.get('openai_api_key')}"},
                   json=body, timeout=float(cfg.get("timeout_seconds", 60)))
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {' '.join(r.text.split())[:300]}")
    out = r.json().get("output", [])
    text = "".join(c.get("text", "") for item in out if item.get("type") == "message"
                   for c in item.get("content", []) if c.get("type") == "output_text")
    return text, out


def _now():
    now = datetime.datetime.now()
    return f"现在是 {now:%Y-%m-%d %H:%M}（星期{WEEKDAYS[now.weekday()]}）。"


def reply(cfg, history, group=False, extra=None):
    """cfg: the [llm] table; extra: more instructions for this one reply. Returns the reply text
    ('' means stay silent)."""
    system = cfg["system_prompt"] + ("\n" + cfg["group_prompt"] if group and cfg.get("group_prompt") else "")
    system += ("\n" + extra if extra else "") + "\n" + _now()
    body = {"model": cfg.get("model", "gpt-5.5"), "instructions": system,
            "input": build_input(history, int(cfg.get("max_images", 3)))}
    if cfg.get("reasoning_effort"):
        body["reasoning"] = {"effort": cfg["reasoning_effort"]}
    if cfg.get("web_search", True):
        tool = {"type": "web_search"}
        if cfg.get("location"):  # e.g. {country = "CN", city = "Beijing"}: "今天天气" means here
            tool["user_location"] = {"type": "approximate", **cfg["location"]}
        body["tools"] = [tool]
    text, out = _post(cfg, body)
    searches = [(i.get("action") or {}).get("query") or "" for i in out if i.get("type") == "web_search_call"]
    if searches:
        log(f"llm web search x{len(searches)}: " + " | ".join(q for q in searches if q)[:200])
    text = _clean(text)
    return "" if SILENT in text else text


def summarize(cfg, scfg, transcript, members=""):
    """Sum up a stretch of group chat (mostly transcribed voices). cfg: [llm], scfg: [summary];
    transcript: plain text lines. Returns the summary text."""
    system = scfg["prompt"] + ("\n\n群成员：\n" + members if members else "") + "\n" + _now()
    body = {"model": scfg.get("model") or cfg.get("model", "gpt-5.5"), "instructions": system,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": transcript}]}]}
    if scfg.get("effort"):
        body["reasoning"] = {"effort": scfg["effort"]}
    text, _ = _post({**cfg, "timeout_seconds": scfg.get("timeout_seconds", cfg.get("timeout_seconds", 60))}, body)
    return _clean(text)


DECISIONS_URL = "https://api.openai.com/v1/decisions"
TO = {"bot": "在跟你说话：叫了你的名字，或接着你上一句追问、纠正你",
      "member": "在跟某个群成员说话：叫了名字或称呼（妈、爸、老公……），回应某人刚说的话，或问只有家里人知道的事",
      "group": "对全群说，没有特定对象",
      "none": "自言自语、通知、寒暄、表态"}
KIND = {"question": "在问问题", "share": "分享生活、照片、见闻", "chat": "闲聊、接话、表态", "other": "其他"}
JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["to", "kind", "reply", "reason"],
    "properties": {
        "to": {"type": "string", "enum": list(TO)},
        "kind": {"type": "string", "enum": list(KIND)},
        "reply": {"type": "boolean"},
        "reason": {"type": "string"},
    },
}


# Decisions API: the model answers facts, the code applies the policy. A single "should you speak up?"
# predicate came back ~0 for every photo share (11-19 replay), while these facts separated cleanly.
FACTS = {
    "needs_response": "新消息需要你回应，不只是“谢谢”“好的”“哈哈”这类收尾。",
    "answerable": "新消息里的问题有确定、客观、你能说准的答案（常识、知识、换算、日期、天气、可以查到的事实），"
                  "不是家里的私事、个人安排或要家人自己定的事。",
    "share_to_group": "新消息是有人在给全群分享生活、照片或见闻，而不是在跟某个人说话、回应某人的问题，也不是转账、通知。",
    "can_comment": "你能针对新消息里的照片或内容，说出一句具体、自然、不客套的话。",
    "unanswered": "新消息里的问题或分享，在新消息里还没有别人回答或接话。",
}
# which facts must all hold, per (to, kind); anything else (to=member, chat, other) stays quiet
POLICY = {"bot": ["needs_response"], "question": ["answerable", "unanswered"],
          "share": ["share_to_group", "can_comment", "unanswered"]}


def judge(cfg, gcfg, transcript, images=(), members=""):
    """Should the bot speak up in a group nobody @-ed it in? cfg: [llm], gcfg: [group].
    transcript: plain text (earlier lines, then the new ones); images: the new burst's images (newest
    few); members: who is who in this group ([group.members]).
    Returns {to, kind, reply, reason, p}. [group] judge_api picks the endpoint:
      decisions  OpenAI Decisions API (gpt-6-luna), ~0.3 s, input-only billing: who is it for, what is it,
                 and the FACTS; reply when every fact POLICY needs for that case has P >= reply_threshold
                 (p = the weakest of them)
      responses  a Responses API call with a JSON schema (judge_model, judge_effort), with a reason"""
    rules = gcfg["judge_prompt"].replace("{names}", "、".join(gcfg.get("names") or ["（未设置）"]))
    if members:
        rules += "\n\n群成员：\n" + members
    if isinstance(images, str):
        images = [images]
    parts = [p for p in (_image_part(i) for i in (images or [])[-3:]) if p]
    content = [{"type": "input_text", "text": transcript}] + [{**p, "detail": "low"} for p in parts]
    tcfg = {**cfg, "timeout_seconds": gcfg.get("judge_timeout_seconds", 30)}
    if gcfg.get("judge_api", "decisions") == "responses":
        body = {"model": gcfg.get("judge_model") or cfg.get("model", "gpt-5.5"),
                "instructions": rules + "\nto 是新消息在跟谁说，kind 是什么类型，reply 是否主动说话，reason 一句话理由。\n" + _now(),
                "input": [{"role": "user", "content": content}],
                "text": {"format": {"type": "json_schema", "name": "decision", "strict": True,
                                    "schema": JUDGE_SCHEMA}}}
        if gcfg.get("judge_effort"):
            body["reasoning"] = {"effort": gcfg["judge_effort"]}
        text, _ = _post(tcfg, body)
        d = json.loads(text)
        return {**{k: d.get(k) for k in JUDGE_SCHEMA["required"]}, "p": None}
    body = {"model": "gpt-6-luna",
            "input": [{"role": "developer", "content": rules + "\n" + _now()},
                      {"role": "user", "content": content}],
            "questions": [
                {"type": "choice", "name": "to", "instructions": "新消息主要在跟谁说话？",
                 "choices": [{"value": k, "description": v} for k, v in TO.items()]},
                {"type": "choice", "name": "kind", "instructions": "新消息主要是什么？",
                 "choices": [{"value": k, "description": v} for k, v in KIND.items()]},
            ] + [{"type": "predicate", "name": k, "instructions": v} for k, v in FACTS.items()]}
    r = httpx.post(DECISIONS_URL, headers={"Authorization": f"Bearer {keys.get('openai_api_key')}"},
                   json=body, timeout=float(tcfg["timeout_seconds"]))
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI decisions HTTP {r.status_code}: {' '.join(r.text.split())[:300]}")
    a = {x.get("name"): x for x in r.json().get("answers", [])}
    if any(a.get(k, {}).get("type") in (None, "refusal") for k in ("to", "kind", *FACTS)):
        raise RuntimeError(f"decisions: missing or refused answer: {a}")
    to, kind = a["to"]["choice"], a["kind"]["choice"]
    facts = {k: float(a[k]["probability"]) for k in FACTS}
    need = [] if to == "member" else POLICY.get("bot" if to == "bot" else kind, [])
    p = min((facts[k] for k in need), default=0.0)
    reason = (f"{TO[to].split('：')[0]}，{KIND[kind]}（to={to} {a['to'].get('confidence', 0):.2f}, "
              f"kind={kind} {a['kind'].get('confidence', 0):.2f}; "
              + " ".join(f"{k}={v:.2f}" for k, v in facts.items() if k in need or not need) + "）")
    return {"to": to, "kind": kind, "reply": bool(need) and p >= float(gcfg.get("reply_threshold", 0.75)),
            "reason": reason, "p": p, "facts": facts}
