"""Command line: run the bot, list chats, or send one text by hand."""
import argparse
import os
import signal
import sys
import threading
import time
import tomllib

from .log import log
from .paths import DATA, ROOT


def _merge(base, over):
    for k, v in over.items():
        base[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def load_config(path):
    """`path`, then `config.local.toml` next to it (gitignored: chats, location, prompts) on top."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    local = os.path.join(os.path.dirname(os.path.abspath(path)), "config.local.toml")
    if os.path.exists(local):
        with open(local, "rb") as f:
            cfg = _merge(cfg, tomllib.load(f))
    return cfg


def run(cfg, dry_run):
    from .pipeline import Pipeline
    from .sender import NativeSender, NotReady
    from .source import Source
    from .store import Store

    os.makedirs(DATA, mode=0o700, exist_ok=True)
    os.chmod(DATA, 0o700)
    store = Store(os.path.join(DATA, "bot.db"))
    pipeline = None
    sender = NativeSender(on_event=lambda ev: pipeline and pipeline.on_sender_event(ev))
    pipeline = Pipeline(cfg, store, sender, dry_run=dry_run)
    source = Source(cfg, store, pipeline.on_message, pipeline.on_wake)
    def stop(*_):  # launchd and the launcher may both deliver SIGTERM; act on the first only
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # Watchdog: before this, a stopped bot outlived its launcher (a leftover thread kept the
        # interpreter alive, and later SIGTERMs were ignored), so every restart left one more
        # copy reading the same chats.
        threading.Thread(target=lambda: (time.sleep(5), os._exit(0)), daemon=True).start()
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    if not dry_run:
        try:
            sender.ensure()  # attach now, so the first manual send after start is captured
        except NotReady as e:
            log(f"sender: {e}; will attach when WeChat-debug writes again")
    log("bot running" + (" (dry run: replies are logged, not sent)" if dry_run else ""))
    try:
        source.run()
    finally:
        sender.close()
        log("bot stopped")
        os._exit(0)  # don't wait for non-daemon threads (Frida, workers): they kept the process alive


def send(to, text, wait):
    from .sender import NativeSender
    s = NativeSender()
    s.ensure()
    try:
        if not s.seed_ready.is_set():
            print("Waiting for one manual text send from WeChat-debug …", flush=True)
            if not s.seed_ready.wait(wait):
                raise SystemExit("no send context captured; nothing sent")
        print(s.send_text(to, text))
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(prog="monitor.py")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.toml"))
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="run the bot (default)")
    r.add_argument("--dry-run", action="store_true", help="log replies instead of sending them")
    c = sub.add_parser("chats", help="list chats")
    c.add_argument("query", nargs="?")
    s = sub.add_parser("send", help="send one text (waits for a manual send as seed)")
    s.add_argument("--to", required=True, help="exact username, e.g. wxid_xxxxxxxx")
    s.add_argument("--text", required=True)
    s.add_argument("--wait", type=int, default=300)
    a = ap.parse_args()
    if a.cmd == "chats":
        from .source import list_chats
        return list_chats(a.query)
    if a.cmd == "send":
        return send(a.to, a.text, a.wait)
    run(load_config(a.config), getattr(a, "dry_run", False))
