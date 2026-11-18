"""Placeholder reply model: an OpenAI chat bot (Responses API) that looks at images and can search the web."""
import base64
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


def reply(cfg, history, group=False):
    """cfg: the [llm] table. Returns the reply text ('' means stay silent)."""
    now = datetime.datetime.now()
    system = cfg["system_prompt"] + ("\n" + cfg["group_prompt"] if group and cfg.get("group_prompt") else "")
    system += f"\n现在是 {now:%Y-%m-%d %H:%M}（星期{WEEKDAYS[now.weekday()]}）。"
    body = {"model": cfg.get("model", "gpt-5.5"), "instructions": system,
            "input": build_input(history, int(cfg.get("max_images", 3)))}
    if cfg.get("reasoning_effort"):
        body["reasoning"] = {"effort": cfg["reasoning_effort"]}
    if cfg.get("web_search", True):
        tool = {"type": "web_search"}
        if cfg.get("location"):  # e.g. {country = "CN", city = "Beijing"}: "今天天气" means here
            tool["user_location"] = {"type": "approximate", **cfg["location"]}
        body["tools"] = [tool]
    r = httpx.post(URL, headers={"Authorization": f"Bearer {keys.get('openai_api_key')}"},
                   json=body, timeout=float(cfg.get("timeout_seconds", 60)))
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI HTTP {r.status_code}: {' '.join(r.text.split())[:300]}")
    out = r.json().get("output", [])
    text = "".join(c.get("text", "") for item in out if item.get("type") == "message"
                   for c in item.get("content", []) if c.get("type") == "output_text")
    searches = [(i.get("action") or {}).get("query") or "" for i in out if i.get("type") == "web_search_call"]
    if searches:
        log(f"llm web search x{len(searches)}: " + " | ".join(q for q in searches if q)[:200])
    return _clean(text)
