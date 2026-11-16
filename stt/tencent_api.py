"""Tencent Cloud ASR (腾讯云 语音识别), base64/raw audio.

  sentence  SentenceRecognition, engine 16k_zh_dialect (23 dialects incl. 上海话; ≤60 s)
  flash     FlashRecognize, engine 16k_zh_en (大模型1.0: 中英粤 + 上海话 and other dialects)
  rec       CreateRecTask + DescribeTaskStatus, engine 16k_zh_en_2.0 (大模型2.0: 31 dialects incl.
            宁波话 and 吴语), polled here
Docs: https://cloud.tencent.com/document/product/1093/35646 , /52097 , /37823
"""
import base64
import datetime
import hashlib
import hmac
import json
import re
import time

import httpx

from . import keys
from .audio import Audio

HOST = "asr.tencentcloudapi.com"


def ready() -> bool:
    return bool(keys.get("tencent_secret_id") and keys.get("tencent_secret_key"))


def ready_flash() -> bool:
    return ready() and bool(keys.get("tencent_app_id"))


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _call(action: str, payload: dict) -> dict:
    """Tencent Cloud API 3.0 with TC3-HMAC-SHA256 signing."""
    sid, skey = keys.get("tencent_secret_id"), keys.get("tencent_secret_key")
    ts = int(time.time())
    date = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%d")
    body = json.dumps(payload)
    ctype = "application/json; charset=utf-8"
    canonical = f"POST\n/\n\ncontent-type:{ctype}\nhost:{HOST}\n\ncontent-type;host\n{_sha256(body)}"
    scope = f"{date}/asr/tc3_request"
    to_sign = f"TC3-HMAC-SHA256\n{ts}\n{scope}\n{_sha256(canonical)}"
    k = _hmac(_hmac(_hmac(("TC3" + skey).encode(), date), "asr"), "tc3_request")
    sig = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    headers = {
        "Authorization": f"TC3-HMAC-SHA256 Credential={sid}/{scope}, SignedHeaders=content-type;host, Signature={sig}",
        "Content-Type": ctype, "Host": HOST,
        "X-TC-Action": action, "X-TC-Version": "2019-06-14", "X-TC-Timestamp": str(ts),
    }
    r = httpx.post(f"https://{HOST}", headers=headers, content=body, timeout=60)
    resp = r.json().get("Response", {})
    if r.status_code != 200 or "Error" in resp:
        err = resp.get("Error", {})
        raise RuntimeError(f"HTTP {r.status_code} {err.get('Code', '')}: {err.get('Message', r.text[:300])}")
    return resp


def sentence(audio: Audio, engine: str = "16k_zh_dialect") -> str:
    if audio.seconds > 60:
        raise ValueError("SentenceRecognition accepts at most 60 s")
    resp = _call("SentenceRecognition", {
        "EngSerViceType": engine, "SourceType": 1, "VoiceFormat": "wav",
        "Data": base64.b64encode(audio.wav).decode(), "DataLen": len(audio.wav),
    })
    return resp["Result"]


def rec(audio: Audio, engine: str = "16k_zh_en_2.0", timeout: float = 120, poll: float = 0.3) -> str:
    resp = _call("CreateRecTask", {
        "EngineModelType": engine, "ChannelNum": 1, "ResTextFormat": 0, "SourceType": 1,
        "Data": base64.b64encode(audio.wav).decode(), "DataLen": len(audio.wav),
    })
    task = resp["Data"]["TaskId"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll)
        d = _call("DescribeTaskStatus", {"TaskId": task})["Data"]
        if d["Status"] == 2:  # success; Result lines look like "[0:0.020,0:2.360]  text"
            return "".join(re.sub(r"^\[[^\]]*\]\s*", "", ln) for ln in d["Result"].splitlines()).strip()
        if d["Status"] == 3:
            raise RuntimeError(f"task failed: {d.get('ErrorMsg', '')}")
    raise TimeoutError("CreateRecTask did not finish in time")


def flash(audio: Audio, engine: str = "16k_zh_en") -> str:
    appid, sid, skey = keys.get("tencent_app_id"), keys.get("tencent_secret_id"), keys.get("tencent_secret_key")
    host, path = "asr.cloud.tencent.com", f"/asr/flash/v1/{appid}"
    params = {"engine_type": engine, "secretid": sid, "timestamp": str(int(time.time())), "voice_format": "wav"}
    query = "&".join(f"{k}={params[k]}" for k in sorted(params))
    sig = base64.b64encode(hmac.new(skey.encode(), f"POST{host}{path}?{query}".encode(), hashlib.sha1).digest()).decode()
    r = httpx.post(f"https://{host}{path}?{query}", content=audio.wav, timeout=120,
                   headers={"Authorization": sig, "Content-Type": "application/octet-stream", "Host": host})
    j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code != 200 or j.get("code") != 0:
        raise RuntimeError(f"HTTP {r.status_code} code {j.get('code')}: {j.get('message', ' '.join(r.text.split())[:200])}")
    return "".join(x["text"] for x in j["flash_result"])
