"""Volcengine 豆包语音 ASR (openspeech.bytedance.com, v3 API).

Three call styles, each selectable with 1.0 (bigasr) or 2.0 (seedasr) resource ids:
  flash   POST .../auc/bigmodel/recognize/flash   one synchronous call, base64 audio
  file    POST .../auc/bigmodel/submit + /query    async task, polled here
  stream  WSS  .../sauc/bigmodel_nostream          binary-framed websocket, whole clip

Dialects: with no `language` set, the bigmodel family auto-handles Mandarin plus
Shanghainese (吴语), Sichuanese, Cantonese, Min Nan, etc. 余姚话 is not listed; test it.
Docs: https://www.volcengine.com/docs/6561/1354868 (file), /1631584 (flash), streaming ASR page.
"""
import base64
import gzip
import json
import struct
import time
import uuid

import httpx
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from . import keys
from .audio import RATE, Audio

HOST = "openspeech.bytedance.com"
OK = "20000000"
PENDING = {"20000001", "20000002"}  # processing, queued


def ready() -> bool:
    return bool(keys.get("volc_api_key") or (keys.get("volc_app_id") and keys.get("volc_access_token")))


def _headers(resource_id: str, request_id: str | None = None, sequence: bool = True) -> dict:
    h = {"X-Api-Resource-Id": resource_id, "X-Api-Request-Id": request_id or str(uuid.uuid4())}
    if sequence:
        h["X-Api-Sequence"] = "-1"
    if keys.get("volc_app_id") and keys.get("volc_access_token"):  # classic console: App ID + Access Token
        h["X-Api-App-Key"] = keys.get("volc_app_id")
        h["X-Api-Access-Key"] = keys.get("volc_access_token")
    else:  # new console: a single key
        h["X-Api-Key"] = keys.get("volc_api_key")
    return h


def _request_opts(extra: dict | None = None) -> dict:
    """`extra` merges into `request`, e.g. {"corpus": {"context": json.dumps({...})}}."""
    return {"model_name": "bigmodel", "enable_itn": True, "enable_punc": True, "enable_ddc": False,
            **(extra or {})}


def _check(r: httpx.Response) -> str:
    code = r.headers.get("X-Api-Status-Code", "")
    if r.status_code != 200 or (code and code != OK and code not in PENDING):
        msg = r.headers.get("X-Api-Message", "") or r.text[:300]
        raise RuntimeError(f"HTTP {r.status_code} status {code}: {msg} (logid {r.headers.get('X-Tt-Logid', '')})")
    return code


# ---- flash (极速版) -----------------------------------------------------------

def flash(audio: Audio, resource_id: str, extra: dict | None = None) -> str:
    body = {
        "user": {"uid": keys.get("volc_app_id") or "wechat-reply"},
        "audio": {"data": base64.b64encode(audio.wav).decode()},
        "request": _request_opts(extra),
    }
    r = httpx.post(f"https://{HOST}/api/v3/auc/bigmodel/recognize/flash",
                   headers=_headers(resource_id), json=body, timeout=120)
    _check(r)
    return r.json()["result"]["text"]


# ---- file (标准版, submit + query) --------------------------------------------

def file(audio: Audio, resource_id: str, timeout: float = 120, extra: dict | None = None,
         poll: float = 0.2) -> str:
    rid = str(uuid.uuid4())
    body = {
        "user": {"uid": keys.get("volc_app_id") or "wechat-reply"},
        # Docs list audio.url; inline base64 `data` is tried here because these clips are local.
        "audio": {"data": base64.b64encode(audio.wav).decode(), "format": "wav",
                  "codec": "raw", "rate": RATE, "bits": 16, "channel": 1},
        "request": _request_opts(extra),
    }
    _check(httpx.post(f"https://{HOST}/api/v3/auc/bigmodel/submit",
                      headers=_headers(resource_id, rid), json=body, timeout=60))
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll)  # short clips finish in ~1 s; default limit is 20 QPS
        r = httpx.post(f"https://{HOST}/api/v3/auc/bigmodel/query",
                       headers=_headers(resource_id, rid, sequence=False), json={}, timeout=60)
        if _check(r) == OK:
            return r.json()["result"]["text"]
    raise TimeoutError("query did not finish in time")


# ---- stream (bigmodel_nostream websocket) ------------------------------------
# 4-byte header: [version|header_size] [msg_type|flags] [serialization|compression] [reserved]

FULL_CLIENT, AUDIO_ONLY, FULL_SERVER, ERROR = 0b0001, 0b0010, 0b1001, 0b1111
NO_SEQ, POS_SEQ, LAST_NO_SEQ, LAST_NEG_SEQ = 0b0000, 0b0001, 0b0010, 0b0011
JSON, GZIP = 0b0001, 0b0001
CHUNK = RATE * 2 // 5  # 200 ms of 16-bit mono PCM


def _frame(msg_type: int, flags: int, serialization: int, payload: bytes) -> bytes:
    body = gzip.compress(payload)
    header = bytes([0x11, (msg_type << 4) | flags, (serialization << 4) | GZIP, 0])
    return header + struct.pack(">I", len(body)) + body


def _parse(data: bytes) -> tuple[int, int, dict]:
    """-> (msg_type, flags, json payload) of one server frame."""
    hsize = (data[0] & 0x0F) * 4
    msg_type, flags = data[1] >> 4, data[1] & 0x0F
    serialization, compression = data[2] >> 4, data[2] & 0x0F
    p = hsize
    if msg_type == ERROR:
        code, size = struct.unpack(">II", data[p:p + 8])
        raise RuntimeError(f"server error {code}: {data[p + 8:p + 8 + size].decode('utf-8', 'replace')}")
    if flags & 0b0001:  # carries a sequence number
        p += 4
    (size,) = struct.unpack(">I", data[p:p + 4])
    payload = data[p + 4:p + 4 + size]
    if compression == GZIP:
        payload = gzip.decompress(payload)
    return msg_type, flags, (json.loads(payload) if serialization == JSON and payload else {})


def stream(audio: Audio, resource_id: str, endpoint: str = "bigmodel_nostream",
           extra: dict | None = None) -> str:
    req = {
        "user": {"uid": keys.get("volc_app_id") or "wechat-reply"},
        "audio": {"format": "pcm", "codec": "raw", "rate": RATE, "bits": 16, "channel": 1},
        "request": _request_opts(extra),
    }
    text = ""
    try:
        ws = connect(f"wss://{HOST}/api/v3/sauc/{endpoint}",
                     additional_headers=_headers(resource_id, sequence=False),
                     max_size=None, open_timeout=15)
    except InvalidStatus as e:
        h = e.response.headers
        detail = h.get("X-Api-Message") or e.response.body.decode("utf-8", "replace")[:300]
        raise RuntimeError(f"HTTP {e.response.status_code} status {h.get('X-Api-Status-Code', '')}: "
                           f"{detail} (logid {h.get('X-Tt-Logid', '')})") from None
    with ws:
        ws.send(_frame(FULL_CLIENT, NO_SEQ, JSON, json.dumps(req).encode()))
        _parse(ws.recv(timeout=15))  # ack for the config frame (raises on error)
        pcm = audio.pcm or b"\0\0"
        chunks = [pcm[i:i + CHUNK] for i in range(0, len(pcm), CHUNK)]
        for i, chunk in enumerate(chunks):
            last = i == len(chunks) - 1
            ws.send(_frame(AUDIO_ONLY, LAST_NO_SEQ if last else NO_SEQ, 0, chunk))
        while True:
            _, flags, msg = _parse(ws.recv(timeout=60))
            text = (msg.get("result") or {}).get("text", text)
            if flags in (LAST_NO_SEQ, LAST_NEG_SEQ):
                return text
