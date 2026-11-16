# SPDX-License-Identifier: GPL-3.0-or-later
"""Resident native text sender (Frida agent inside ~/Applications/WeChat-debug.app).

Attaches lazily on the first send and re-attaches after WeChat restarts (Frida reports the
detach; nothing polls the process). Sending needs a seed: the agent clones one from any text
WeChat itself sends, so after each WeChat launch one manual text send is required.
"""
import json
import secrets
import subprocess
import threading
import time
from pathlib import Path

import frida

from ..log import log

AGENT = Path(__file__).with_name("agent.js")
DEBUG_EXECUTABLE = str(Path.home() / "Applications/WeChat-debug.app/Contents/MacOS/WeChat")


class NotReady(RuntimeError):
    """No usable send context yet; the message was not submitted."""


def _varint(v: int) -> bytes:
    out = bytearray()
    while v > 127:
        out.append((v & 127) | 128)
        v >>= 7
    out.append(v)
    return bytes(out)


def _field(n: int, v) -> bytes:
    if isinstance(v, int):
        return _varint(n << 3) + _varint(v)
    return _varint((n << 3) | 2) + _varint(len(v)) + v


def text_proto(to: str, text: str, msg_id: int) -> bytes:
    body = (_field(1, _field(1, to.encode())) + _field(2, text.encode()) + _field(3, 1) +
            _field(4, int(time.time())) + _field(5, msg_id) +
            _field(6, b"<msgsource><alnode><fr>1</fr></alnode></msgsource>"))
    return _field(1, 1) + _field(2, body)


def debug_pid() -> int | None:
    rows = subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True).splitlines()
    pids = [int(p) for row in rows if len(bits := row.strip().split(None, 1)) == 2
            for p, exe in [bits] if exe == DEBUG_EXECUTABLE]
    if len(pids) > 1:
        raise RuntimeError(f"{len(pids)} WeChat-debug main processes")
    return pids[0] if pids else None


class NativeSender:
    def __init__(self, on_event=None):
        self.on_event = on_event or (lambda ev: None)
        self.lock = threading.Lock()
        self.session = self.script = None
        self.pid = None
        self.seed_ready = threading.Event()

    # ---- attachment ----------------------------------------------------------
    def _attach(self):
        pid = debug_pid()
        if pid is None:
            raise NotReady("WeChat-debug is not running")
        session = frida.attach(pid)
        script = session.create_script(AGENT.read_text())
        script.on("message", self._on_message)
        session.on("detached", lambda reason, crash: self._on_detached(pid, reason))
        script.load()
        script.exports_sync.initialize()
        self.session, self.script, self.pid = session, script, pid
        self.seed_ready.clear()
        log(f"sender: attached to WeChat-debug pid {pid}; waiting for one manual text send as seed")

    def _on_detached(self, pid, reason):
        log(f"sender: detached from pid {pid} ({reason})")
        with self.lock:
            if self.pid == pid:
                self.session = self.script = self.pid = None
                self.seed_ready.clear()

    def _on_message(self, message, _data):
        ev = message.get("payload") if message.get("type") == "send" else message
        if isinstance(ev, dict) and ev.get("type") == "seed_ready":
            self.seed_ready.set()
            log(f"sender: send context captured (#{ev.get('seedCount')})")
        elif message.get("type") == "error":
            log(f"sender agent error: {message.get('description')}")
        self.on_event(ev)

    def ensure(self):
        """Attach if needed. Raises NotReady if WeChat-debug is not running."""
        with self.lock:
            if self.script is None:
                self._attach()

    def status(self) -> dict:
        with self.lock:
            if self.script is None:
                return {"attached": False}
            return {"attached": True, **self.script.exports_sync.status()}

    def close(self):
        """Detach. If WeChat still holds tasks that call our serializer, leave the agent in place."""
        with self.lock:
            if self.script is None:
                return
            try:
                if self.script.exports_sync.status()["inFlight"]:
                    self.script.eternalize()
                    log("sender: tasks in flight; agent stays loaded until WeChat exits")
                self.session.detach()
            except Exception as e:
                log(f"sender: close failed: {e}")
            self.session = self.script = self.pid = None

    # ---- sending -------------------------------------------------------------
    def send_text(self, to: str, text: str) -> dict:
        """Submit once (never retried). Raises NotReady when nothing was submitted.

        A returned result means WeChat accepted the task, not that the recipient got it.
        """
        if not to or not text.strip() or len(text.encode()) > 16000:
            raise ValueError("recipient and 1..16000 bytes of text required")
        self.ensure()
        if not self.seed_ready.is_set():
            raise NotReady("no send context yet: send any text from WeChat-debug once")
        msg_id = (1 << 34) | secrets.randbits(34)
        with self.lock:
            if self.script is None:
                raise NotReady("WeChat-debug went away")
            try:
                r = self.script.exports_sync.submit(msg_id, list(text_proto(to, text, msg_id)))
            except frida.core.RPCException as e:
                if "no send context" in str(e) or "not initialized" in str(e):
                    raise NotReady(str(e)) from e
                raise
        return {"clientMessageId": msg_id, **r}


def main():
    """Experiment: one seed, several sends spaced out in time.

    uv run -m wxbot.sender.native --to <username> --at 0,60,360 --tag WXTEST-1114-03
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True)
    ap.add_argument("--tag", default="WXTEST")
    ap.add_argument("--at", default="0,60,360", help="seconds after the seed to send each message")
    ap.add_argument("--seed-timeout", type=int, default=300)
    a = ap.parse_args()
    s = NativeSender(on_event=lambda ev: print(json.dumps(ev, ensure_ascii=False), flush=True))
    s.ensure()
    import atexit
    atexit.register(s.close)
    print("Waiting for one manual text send from WeChat-debug …", flush=True)
    if not s.seed_ready.wait(a.seed_timeout):
        raise SystemExit("no seed captured; nothing sent")
    t0 = time.monotonic()
    for i, delay in enumerate(int(x) for x in a.at.split(",")):
        time.sleep(max(0, t0 + delay - time.monotonic()))
        text = f"【{a.tag}-{chr(97 + i)}】第 {i + 1} 条，种子后 {delay} 秒发送"
        try:
            print(json.dumps({"send": i + 1, "text": text, **s.send_text(a.to, text)}, ensure_ascii=False), flush=True)
        except Exception as e:
            print(json.dumps({"send": i + 1, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False), flush=True)
    time.sleep(10)
    print(json.dumps({"final": s.status()}), flush=True)


if __name__ == "__main__":
    main()
