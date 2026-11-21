"""One structured LLM call (JSON with fixed string fields), shared by the labeling page and labeling/sources/."""
import base64
import json
import subprocess
import tempfile

import httpx

from stt import keys


def llm_json(model: str, prompt: str, fields: tuple[str, ...], wav: str | None = None) -> dict:
    """One structured LLM call returning the string `fields`. Gemini can also get a WAV clip; GPT is text only.
    "agy:<model name>" runs the Antigravity CLI instead (`agy -p … --model …`, text only, no API key)."""
    if model.startswith("agy:"):
        return _agy_json(model.removeprefix("agy:"), prompt, fields, wav)
    if model.startswith("gemini"):
        parts = [{"inline_data": {"mime_type": "audio/wav",
                                  "data": base64.b64encode(open(wav, "rb").read()).decode()}}] if wav else []
        parts.append({"text": prompt})
        schema = {"type": "OBJECT", "properties": {k: {"type": "STRING"} for k in fields}, "required": list(fields)}
        r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                       headers={"x-goog-api-key": keys.get("gemini_api_key")}, timeout=180,
                       json={"contents": [{"role": "user", "parts": parts}],
                             "generationConfig": {"responseMimeType": "application/json", "responseSchema": schema}})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {' '.join(r.text.split())[:300]}")
        cands = r.json().get("candidates") or []
        text = "".join(p.get("text", "") for p in (cands[0].get("content", {}).get("parts", []) if cands else [])
                       if not p.get("thought"))
    else:
        if wav:
            raise ValueError(f"{model} can't take audio")
        schema = {"type": "object", "properties": {k: {"type": "string"} for k in fields},
                  "required": list(fields), "additionalProperties": False}
        r = httpx.post("https://api.openai.com/v1/chat/completions",
                       headers={"Authorization": f"Bearer {keys.get('openai_api_key')}"}, timeout=180,
                       json={"model": model, "messages": [{"role": "user", "content": prompt}],
                             "response_format": {"type": "json_schema",
                                                 "json_schema": {"name": "label", "strict": True, "schema": schema}}})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {' '.join(r.text.split())[:300]}")
        text = r.json()["choices"][0]["message"]["content"]
    out = json.loads(text)
    return {k: str(out.get(k, "")).strip() for k in fields}


def _agy_json(model: str, prompt: str, fields: tuple[str, ...], wav: str | None) -> dict:
    if wav:
        raise ValueError("agy is used text only")
    schema = {"type": "object", "properties": {k: {"type": "string"} for k in fields}, "required": list(fields)}
    with tempfile.TemporaryDirectory() as d:  # empty workspace: nothing for the agent to read
        r = subprocess.run(["agy", "-p", prompt, "--model", model, "--json-schema", json.dumps(schema),
                            "--output-format", "json", "--print-timeout", "300s", "--sandbox"],
                           cwd=d, capture_output=True, text=True, timeout=360)
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"agy: {(r.stderr or r.stdout).strip()[:300]}")
    if out.get("status") != "SUCCESS" or not isinstance(out.get("structured_output"), dict):
        raise RuntimeError(f"agy: {out.get('status')} {str(out.get('response'))[:300]}")
    return {k: str(out["structured_output"].get(k, "")).strip() for k in fields}
