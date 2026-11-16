"""Per-chat reply pipeline: queue -> understand -> debounce -> LLM -> send.

One worker thread per chat keeps that chat in order without blocking the others. Everything is
driven by incoming messages; the only waits are the debounce window and bounded API calls.
"""
import os
import queue
import re
import subprocess
import threading
import time

from . import llm
from .log import log
from .paths import ROOT
from .sender import NotReady
from .understand import understand

MENTION = re.compile(r"@[^\s\u2005]+[\s\u2005]*")


class Pipeline:
    def __init__(self, cfg, store, sender, dry_run=False):
        self.cfg, self.store, self.sender, self.dry_run = cfg, store, sender, dry_run
        self.r = cfg.get("reply", {})
        self.workers = {}
        self.pending = {}  # chat -> (created, text): replies held until a send context exists
        self.lock = threading.Lock()
        self.last_notice = 0.0
        self.last_attach_try = 0.0

    # ---- inputs ------------------------------------------------------------
    def on_message(self, ev):
        """From the source worker thread, for every message in a watched chat."""
        if ev["is_self"]:  # manual sends (e.g. the seed) are not part of the conversation
            return
        if ev.get("reply", "always") == "never":  # listen-only chat: recorded by the source, never answered
            return
        with self.lock:
            w = self.workers.get(ev["chat"])
            if w is None:
                w = self.workers[ev["chat"]] = queue.Queue()
                threading.Thread(target=self._chat_loop, args=(ev["chat"], w), name=f"chat-{ev['chat']}",
                                 daemon=True).start()
        w.put(ev)

    def on_wake(self):
        """Source saw a DB write. If WeChat-debug came back, attach so the next manual send is captured."""
        if self.dry_run or self.sender.status().get("attached"):
            return
        if time.time() - self.last_attach_try < 10:
            return
        self.last_attach_try = time.time()
        try:
            self.sender.ensure()
            self._notice("微信机器人已连接，请在调试版微信里手动发一条文字以启用回复")
        except NotReady:
            pass
        except Exception as e:
            log(f"sender attach failed: {type(e).__name__}: {e}")

    def on_sender_event(self, ev):
        """From Frida's message thread; never call into the agent from here."""
        if isinstance(ev, dict) and ev.get("type") == "seed_ready":
            threading.Thread(target=self._flush_pending, name="flush", daemon=True).start()

    # ---- per-chat worker ---------------------------------------------------
    def _chat_loop(self, chat, q):
        debounce = float(self.r.get("debounce_seconds", 2))
        while True:
            batch = [q.get()]
            self._ingest(chat, batch[0])
            while True:  # a burst of messages gets one reply
                try:
                    ev = q.get(timeout=debounce)
                except queue.Empty:
                    break
                batch.append(ev)
                self._ingest(chat, ev)
            try:
                self._respond(chat, batch)
            except Exception as e:
                log(f"[{chat}] reply failed: {type(e).__name__}: {e}")

    def _ingest(self, chat, ev):
        try:
            # mention-only chats: voice stays on this machine (placeholder, no cloud STT)
            stt = None if ev.get("reply") == "mention" else self.cfg.get("stt", {}).get("provider", "combo")
            seen = understand(ev, stt)
        except Exception as e:
            log(f"[{chat}] understand {ev['local_id']} failed: {type(e).__name__}: {e}")
            seen = (f"[{ev['type']}]", None)
        if seen is None:
            ev["_skip"] = True
            return
        text, image = seen
        if ev.get("is_group"):  # who said it, for the LLM
            text = f"{ev.get('sender_name') or ev.get('sender')}：{text}"
        self.store.add(chat, "user", text, image, at=ev["time"], local_id=ev["local_id"], type=ev["type"])
        log(f"[{chat}] <- {text[:80]!r}" + (" +image" if image else ""))

    def _respond(self, chat, batch):
        live = [ev for ev in batch if not ev.get("_skip")]
        if not live:
            return
        now = time.time()
        age = now - live[-1]["time"]
        if self._command(chat, live, fresh=age <= float(self.r.get("max_age_seconds", 600))):
            return
        if self.store.muted_until(chat) > now:  # "@<me> 关闭": record only, no LLM
            log(f"[{chat}] muted until {time.strftime('%m-%d %H:%M', time.localtime(self.store.muted_until(chat)))}; not replying")
            return
        if live[-1].get("reply") == "mention" and not any(ev.get("mentions_me") for ev in live):
            return  # recorded as context; answered only when this account is @-mentioned
        if age > float(self.r.get("max_age_seconds", 600)):
            log(f"[{chat}] newest message is {age:.0f}s old (backlog); not replying")
            return
        if os.path.exists(os.path.join(ROOT, self.r.get("stop_file", "data/STOP"))):
            log(f"[{chat}] stop file present; not replying")
            return
        if self.store.replies_since(chat, now - 60) >= int(self.r.get("max_per_minute", 6)):
            log(f"[{chat}] rate limit reached; not replying")
            return
        # context starts after an "@<me> 打开" or, once a mute has run out, at its end: nothing said
        # before or during a mute reaches the LLM (a mute still running never gets here)
        since = max(now - float(self.r.get("history_hours", 12)) * 3600, self.store.history_floor(chat),
                    self.store.muted_until(chat))
        history = self.store.history(chat, since)
        t = time.monotonic()
        text = llm.reply(self.cfg["llm"], history, group=live[-1].get("is_group", False))
        log(f"[{chat}] llm {time.monotonic() - t:.1f}s -> {text[:80]!r}")
        if text:
            self._deliver(chat, text)

    def _command(self, chat, live, fresh):
        """"@<me> 关闭" mutes the chat for mute_hours (from the message's time, so it holds across a
        restart or backlog); "@<me> 打开" unmutes. The confirmation is a fixed text, not the LLM.
        Returns True when the batch's last command was handled."""
        cmd = None
        for ev in live:
            if not ev.get("mentions_me"):
                continue
            word = MENTION.sub("", ev.get("text") or "").strip(" 。.!！~～")
            if word in self.r.get("mute_words", ["关闭"]):
                cmd = ("off", ev)
            elif word in self.r.get("unmute_words", ["打开", "开启"]):
                cmd = ("on", ev)
        if not cmd:
            return False
        kind, ev = cmd
        hours = float(self.r.get("mute_hours", 24))
        if kind == "off":
            until = ev["time"] + hours * 3600
            self.store.set_mute(chat, until)
            log(f"[{chat}] muted by {ev.get('sender_name')} until {time.strftime('%m-%d %H:%M', time.localtime(until))}")
            ack = f"好的，接下来{hours:g}小时我只看不说。想提前叫醒我就 @我 说「打开」。"
        else:
            if self.store.muted_until(chat) <= time.time():
                return False  # not muted: let the LLM answer as usual
            self.store.set_mute(chat, 0)
            # start the LLM's context over: nothing from before (or of) the 打开 message; WeChat times
            # are whole seconds, the confirmation below is stored with a later float time
            self.store.set_history_floor(chat, ev["time"] + 0.5)
            log(f"[{chat}] unmuted by {ev.get('sender_name')}; history starts now")
            ack = "好的，我回来了。"
        if fresh and not os.path.exists(os.path.join(ROOT, self.r.get("stop_file", "data/STOP"))):
            self._deliver(chat, ack)
        return True

    # ---- sending -----------------------------------------------------------
    def _deliver(self, chat, text):
        if self.dry_run:
            log(f"[{chat}] (dry run) would send {text!r}")
            return
        try:
            r = self.sender.send_text(chat, text)
        except NotReady as e:
            with self.lock:
                self.pending[chat] = (time.time(), text)
            log(f"[{chat}] held: {e}")
            self._notice("微信机器人有待发的回复：请在调试版微信里手动发一条文字")
            return
        except Exception as e:  # ambiguous: never retry, a duplicate is worse than a gap
            log(f"[{chat}] send failed, not retried: {type(e).__name__}: {e}")
            return
        self.store.add(chat, "assistant", text, client_msg_id=r["clientMessageId"], task_id=r.get("taskId"))
        log(f"[{chat}] -> sent (task {r.get('taskId')}, result {r.get('result')})")

    def _flush_pending(self):
        ttl = float(self.r.get("pending_ttl_seconds", 300))
        with self.lock:
            items, self.pending = self.pending, {}
        for chat, (created, text) in items.items():
            if time.time() - created > ttl:
                log(f"[{chat}] dropped a held reply ({time.time() - created:.0f}s old)")
                continue
            self._deliver(chat, text)

    def _notice(self, message):
        if time.time() - self.last_notice < 600:
            return
        self.last_notice = time.time()
        script = "on run argv\ndisplay notification (item 1 of argv) with title \"WeChat bot\"\nend run"
        subprocess.run(["osascript", "-e", script, message], check=False, capture_output=True)
