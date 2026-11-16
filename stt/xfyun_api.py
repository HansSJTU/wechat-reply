"""iFlytek 讯飞 语音听写大模型（中英）V2.0 — websocket, 16 kHz PCM, ≤60 s.

Claims 202 Chinese dialects with no selector (domain spark_asr_v2); the published list includes
宁波话, 上海话, 杭州话, 绍兴话, 苏州话, 温州话, 台州话, 舟山话.
Docs: https://www.xfyun.cn/doc/spark/spark_zh_iat_new.html
"""
import base64
import hashlib
import hmac
import json
import time
from email.utils import formatdate
from urllib.parse import urlencode

from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

from . import keys
from .audio import RATE, Audio

HOST, PATH = "iat.cn-huabei-1.xf-yun.com", "/v1"
FRAME = 8000  # bytes of PCM per frame (250 ms); docs suggest 40 ms frames for live audio


def ready() -> bool:
    return all(keys.get(k) for k in ("xf_app_id", "xf_api_key", "xf_api_secret"))


def _url() -> str:
    date = formatdate(usegmt=True)
    origin = f"host: {HOST}\ndate: {date}\nGET {PATH} HTTP/1.1"
    sig = base64.b64encode(hmac.new(keys.get("xf_api_secret").encode(), origin.encode(), hashlib.sha256).digest()).decode()
    auth = (f'api_key="{keys.get("xf_api_key")}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{sig}"')
    q = urlencode({"authorization": base64.b64encode(auth.encode()).decode(), "date": date, "host": HOST})
    return f"wss://{HOST}{PATH}?{q}"


def transcribe(audio: Audio, *args, **kw) -> str:
    try:
        return _transcribe(audio, *args, **kw)
    except ConnectionClosed as e:  # server closes with the error text as the close reason
        reason = e.rcvd.reason if e.rcvd else str(e)
        raise RuntimeError(f"closed: {reason}" + (" (no quota / concurrency limit)" if "licc" in reason else "")) from None


def _transcribe(audio: Audio, domain: str = "spark_asr_v2", accent: str = "mandarin", eos: int | None = None, pace: float = 0.0) -> str:
    """domain spark_asr_v2 = 语音听写大模型（中英）V2.0; domain slm + accent mulacc = 方言大模型
    (普通话 + 202 dialects, auto-detected; https://www.xfyun.cn/doc/spark/spark_slm_iat.html)."""
    if audio.seconds > 60:
        raise ValueError("iFlytek dictation accepts at most 60 s")
    app_id = keys.get("xf_app_id")
    pcm = audio.pcm or b"\0\0"
    chunks = [pcm[i:i + FRAME] for i in range(0, len(pcm), FRAME)]
    try:
        ws = connect(_url(), max_size=None, open_timeout=15)
    except InvalidStatus as e:
        raise RuntimeError(f"HTTP {e.response.status_code}: {e.response.body.decode('utf-8', 'replace')[:300]}") from None
    pieces: dict[int, str] = {}
    with ws:
        for i, chunk in enumerate(chunks):
            status = 0 if i == 0 else (2 if i == len(chunks) - 1 else 1)
            if len(chunks) == 1:
                status = 0
            msg = {
                "header": {"app_id": app_id, "status": status},
                "payload": {"audio": {"encoding": "raw", "sample_rate": RATE, "channels": 1, "bit_depth": 16,
                                      "seq": i + 1, "status": status,
                                      "audio": base64.b64encode(chunk).decode()}},
            }
            if i == 0:
                msg["parameter"] = {"iat": {"domain": domain, "language": "zh_cn", "accent": accent,
                                            "result": {"encoding": "utf8", "compress": "raw", "format": "json"}}}
                if eos:
                    msg["parameter"]["iat"]["eos"] = eos
            ws.send(json.dumps(msg))
            if pace:
                time.sleep(pace)
        if len(chunks) == 1:  # a single chunk still needs a closing frame
            ws.send(json.dumps({"header": {"app_id": app_id, "status": 2},
                                "payload": {"audio": {"encoding": "raw", "sample_rate": RATE, "channels": 1,
                                                      "bit_depth": 16, "seq": 2, "status": 2, "audio": ""}}}))
        while True:
            m = json.loads(ws.recv(timeout=60))
            h = m.get("header", {})
            if h.get("code", 0) != 0:
                raise RuntimeError(f"code {h.get('code')}: {h.get('message', '')}")
            res = (m.get("payload") or {}).get("result")
            if res and res.get("text"):
                r = json.loads(base64.b64decode(res["text"]))
                if r.get("pgs") == "rpl":
                    for sn in range(r["rg"][0], r["rg"][1] + 1):
                        pieces.pop(sn, None)
                pieces[r["sn"]] = "".join(w["cw"][0]["w"] for w in r.get("ws", []))
            if h.get("status") == 2:
                return "".join(pieces[k] for k in sorted(pieces))
