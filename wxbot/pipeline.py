"""Per-chat reply pipeline: queue -> understand -> debounce -> LLM -> send.

One worker thread per chat keeps that chat in order without blocking the others. Everything is
driven by incoming messages; the only waits are the debounce window and bounded API calls.

Each watched chat is switched on or off (persisted in data/bot.db): in a 1:1 chat by sending 开启 / 关闭,
in a group by "@<me> 开启" / "@<me> 关闭". Before the first switch, chats listed under `chats` / `mention`
start on and `listen` ones off.
  on   voices are transcribed (stt.provider, e.g. combo: local ASR → Mandarin via an LLM); every candidate
       and the Mandarin go to data/voice_stt.jsonl, the Mandarin into the conversation. 1:1 chats are
       answered every time. Groups: an @ of this account (or a quoted reply to it) is always answered;
       with [group] auto the bot may also speak up unasked, but sparingly (see _judge).
  off  nothing leaves the machine: no STT, no LLM, no reply; messages are only recorded.
Either way every voice is saved (data/media, messages.jsonl) and the labeling page drafts it with the local
models.
"""
import json
import os
import queue
import re
import subprocess
import threading
import time

from . import llm
from .log import log
from .paths import DATA, ROOT
from .sender import NotReady
from .understand import understand

try:
    from stt import PROVIDERS as STT_PROVIDERS, transcribe as stt_transcribe
except ImportError:  # pragma: no cover
    STT_PROVIDERS, stt_transcribe = {}, None

MENTION = re.compile(r"@[^\s\u2005]+[\s\u2005]*")
FILLER = " \u2005。.!！~～，,、？?：:"  # around a spoken 开启 / 关闭
# what the reply model is told about why it is speaking up unasked (the Decisions API gives no reason text)
AUTO_HINT = {"bot": "有人没 @ 你但在跟你说话，回应他。",
             "question": "新消息里有个对大家提的问题，还没人回答，你能答准，就答这个问题。",
             "share": "有人在分享生活、照片，还没人接话：接一句具体的——说出照片里的东西、接个梗，或问一句相关的；别泛泛夸“真好看”。",
             "chat": "群里在闲聊，你有一句真正有用的话可说。",
             "other": ""}
DECISIONS = os.path.join(DATA, "group_decisions.jsonl")  # every unprompted-turn verdict, for tuning


class Pipeline:
    def __init__(self, cfg, store, sender, dry_run=False):
        self.cfg, self.store, self.sender, self.dry_run = cfg, store, sender, dry_run
        self.r = cfg.get("reply", {})
        self.g = cfg.get("group", {})
        self.s = cfg.get("summary", {})
        self.summarized = {}  # chat -> time of the last summary made (also covers dry runs and held sends)
        self.workers = {}
        self.pending = {}  # chat -> (created, text, meta): replies held until a send context exists
        self.lock = threading.Lock()
        self.last_notice = 0.0
        self.last_attach_try = 0.0

    # ---- inputs ------------------------------------------------------------
    def on_message(self, ev):
        """From the source worker thread, for every message in a watched chat."""
        if ev["is_self"]:  # manual sends (e.g. the seed) are not part of the conversation
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

    def _on(self, chat, ev):
        s = self.store.enabled(chat)
        return ev.get("reply") in ("always", "mention") if s is None else s

    def _spot(self, chat, ev):
        """关闭: listen for a spoken 开启 with a local-only model (the first ready one of [reply] command_stt).
        The transcript is used for that check alone: not stored, not logged, not sent anywhere."""
        wanted = self.r.get("command_stt", ["whisper-ft", "wu-ft"])
        wanted = [wanted] if isinstance(wanted, str) else wanted
        provider = next((p for p in wanted if p in STT_PROVIDERS and STT_PROVIDERS[p].ready()), None)
        if not provider or stt_transcribe is None:
            log(f"[{chat}] command spotting: none of {wanted} is ready")
            return None
        r = stt_transcribe(ev["voice_path"], provider)
        if r.error:
            log(f"[{chat}] command spotting ({provider}) failed: {r.error}")
        return f"[语音] {r.text}" if r.text else None

    @staticmethod
    def _addressed(ev):
        return bool(ev.get("mentions_me") or ev.get("quotes_me"))

    def _group_wait(self, chat, batch):
        """An unprompted group turn waits `wait_seconds` after the last message instead of the debounce:
        people get the first chance to answer each other, and what they say lands in the same batch."""
        ev = batch[-1]
        if (not ev.get("is_group") or not self.g.get("auto") or any(self._addressed(e) for e in batch)
                or time.time() - ev["time"] > float(self.r.get("max_age_seconds", 600))
                or not self._on(chat, ev)):
            return 0
        return float(self.g.get("wait_seconds", 20))

    # ---- per-chat worker ---------------------------------------------------
    def _chat_loop(self, chat, q):
        debounce = float(self.r.get("debounce_seconds", 2))
        last = None  # newest event of this chat: is it a group, is it on
        while True:
            # idle: the pending voice summary, if any, is due [summary] wait_seconds after the newest voice
            try:
                due = self._summary_deadline(chat, last)
            except Exception as e:  # the add-on must never take the chat's replies down with it
                log(f"[{chat}] summary check failed: {type(e).__name__}: {e}")
                due = None
            try:
                batch = [q.get(timeout=None if due is None else max(0.0, due - time.time()))]
            except queue.Empty:
                try:
                    self._auto_summary(chat, last)
                except Exception as e:
                    log(f"[{chat}] summary failed: {type(e).__name__}: {e}")
                    self.summarized[chat] = time.time()
                continue
            self._ingest(chat, batch[0])
            while True:  # a burst of messages gets one reply
                try:
                    ev = q.get(timeout=self._group_wait(chat, batch) or debounce)
                except queue.Empty:
                    break
                batch.append(ev)
                self._ingest(chat, ev)
            last = batch[-1]
            try:
                self._respond(chat, batch)
            except Exception as e:
                log(f"[{chat}] reply failed: {type(e).__name__}: {e}")

    def _ingest(self, chat, ev):
        try:
            # 关闭: the voice stays on this machine (placeholder, no STT)
            on = self._on(chat, ev)
            stt = self.cfg.get("stt", {}).get("provider", "combo") if on else None
            seen = understand(ev, stt)
            if not on and ev["type"] == "voice" and ev.get("voice_path"):
                ev["_heard"] = self._spot(chat, ev)
        except Exception as e:
            log(f"[{chat}] understand {ev['local_id']} failed: {type(e).__name__}: {e}")
            seen = (f"[{ev['type']}]", None)
        if seen is None:
            ev["_skip"] = True
            return
        text, image = seen
        ev["_text"] = text
        if ev.get("is_group") and ev["type"] == "voice" and any(n in text for n in self.g.get("names") or []):
            ev["mentions_me"] = True  # nobody can @ in a voice: saying the bot's name counts as one
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
        if not self._on(chat, live[-1]):  # 关闭: record only, no LLM
            return
        group = live[-1].get("is_group", False)
        addressed = not group or any(self._addressed(ev) for ev in live)
        if not addressed and not self.g.get("auto"):
            return  # group: recorded as context; answered only when this account is @-mentioned
        if age > float(self.r.get("max_age_seconds", 600)):
            log(f"[{chat}] newest message is {age:.0f}s old (backlog); not replying")
            return
        if os.path.exists(os.path.join(ROOT, self.r.get("stop_file", "data/STOP"))):
            log(f"[{chat}] stop file present; not replying")
            return
        if self.store.replies_since(chat, now - 60) >= int(self.r.get("max_per_minute", 6)):
            log(f"[{chat}] rate limit reached; not replying")
            return
        # context starts at the last 开启: nothing said while the chat was off reaches the LLM
        since = max(now - float(self.r.get("history_hours", 12)) * 3600, self.store.history_floor(chat))
        history = self.store.history(chat, since)
        extra, auto = None, None
        if group and addressed:
            extra = self.g.get("mention_prompt")
        elif group:
            d = self._judge(chat, live, history, now)
            if not d:
                return
            extra = self.g.get("auto_prompt", "").replace("{reason}", AUTO_HINT.get(
                "bot" if d["to"] == "bot" else d["kind"], "") + (f"（{d['reason']}）" if d.get("p") is None else ""))
            auto = d["kind"] if d["to"] != "bot" else "bot"
        if group and self._members(chat):
            extra = "群成员：\n" + self._members(chat) + ("\n" + extra if extra else "")
        t = time.monotonic()
        text = llm.reply(self.cfg["llm"], history, group=group, extra=extra)
        log(f"[{chat}] llm {time.monotonic() - t:.1f}s -> {text[:80]!r}" + (f" (unprompted, {auto})" if auto else ""))
        if text:
            self._deliver(chat, text, auto=auto)
        elif auto:
            self._record(chat, live, None, "reply model backed out")

    def _members(self, chat):
        """[group.members] "<chat>": who is who in this group (names as the messages show them)."""
        return (self.g.get("members") or {}).get(chat, "").strip()

    # ---- voice chat summaries ----------------------------------------------
    def _summary_since(self, chat, now):
        """A summary covers what came after the last one, the last 开启, and at most [summary] max_hours."""
        return max(now - float(self.s.get("max_hours", 12)) * 3600, self.store.history_floor(chat),
                   self.store.last_summary(chat), self.summarized.get(chat, 0.0))

    def _summary_deadline(self, chat, ev):
        """[summary] auto, group 开启, at least `min_voices` transcribed voices since the last summary: the
        summary is due `wait_seconds` after the newest voice (texts in between don't push it back).
        None when nothing is pending, including voices too old to still be worth it (backlog after downtime)."""
        if not ev or not ev.get("is_group") or not self.s.get("auto", True) or not self._on(chat, ev):
            return None
        n, newest = self.store.voices(chat, self._summary_since(chat, time.time()))
        if n < int(self.s.get("min_voices", 5)):
            return None
        due = newest + float(self.s.get("wait_seconds", 600))
        return None if time.time() - due > float(self.r.get("max_age_seconds", 600)) else due

    def _auto_summary(self, chat, ev):
        """The summary timer went off: post it unless the bot is paused. A skipped one is not retried."""
        now = time.time()
        if self._summary_deadline(chat, ev) is None:  # switched off meanwhile, or a manual one covered it
            return
        stop = os.path.exists(os.path.join(ROOT, self.r.get("stop_file", "data/STOP")))
        busy = self.store.replies_since(chat, now - 60) >= int(self.r.get("max_per_minute", 6))
        if stop or busy:
            log(f"[{chat}] summary skipped: {'stop file present' if stop else 'rate limit reached'}")
            self.summarized[chat] = now
            return
        self._summarize(chat, now)

    def _summarize(self, chat, now, manual=False):
        """Post a summary of the group chat since the last one (mostly transcribed voices)."""
        since = self._summary_since(chat, now)
        rows = self.store.history(chat, since)
        said = [h for h in rows if h["role"] == "user"]
        if len(said) < (2 if manual else 1):  # manual: the request itself is one of them
            if manual:
                self._deliver(chat, "上次总结之后还没有新的聊天，暂时没什么可总结的。")
            return
        voices, _ = self.store.voices(chat, since)
        me = (self.g.get("names") or ["机器人"])[0]
        hm = lambda t: time.strftime("%H:%M", time.localtime(t))
        transcript = "\n".join(f"[{hm(h['time'])}] " + (f"{me}（机器人）：" if h["role"] == "assistant" else "")
                               + h["text"] for h in rows)
        # the next summary starts after the newest line covered here (not at send time: voices that land while
        # the LLM writes belong to the next one); set even if the LLM or the send fails, so no retry storm
        upto = rows[-1]["time"] + 0.001  # history() is inclusive
        self.summarized[chat] = upto
        t = time.monotonic()
        try:
            text = llm.summarize(self.cfg["llm"], self.s, transcript, self._members(chat))
        except Exception as e:
            log(f"[{chat}] summary failed: {type(e).__name__}: {e}")
            return
        log(f"[{chat}] summary of {len(said)} messages ({voices} voices) {time.monotonic() - t:.1f}s "
            f"-> {text[:80]!r}" + (" (asked)" if manual else ""))
        if not text:
            return
        head = (f"【聊天总结】{hm(said[0]['time'])}–{hm(said[-1]['time'])}，"
                + (f"{voices} 条语音" if voices else f"{len(said)} 条消息"))
        self._deliver(chat, head + "\n" + text, summary=voices, upto=upto)

    # ---- unprompted group turns --------------------------------------------
    @staticmethod
    def _eligible(ev):
        """Could this message, on its own, be worth an unprompted word? Free filter before the judge."""
        kind, text = ev.get("type"), ev.get("_text") or ""  # what understand() made of it, no "name：" prefix
        if kind == "text":
            return not ev.get("at_others") and bool(text.strip())
        if kind == "voice":  # transcribed only: "[语音] …", not "[语音]" / "[语音，无法识别]"
            return text.startswith("[语音] ") and bool(text[len("[语音] "):].strip())
        if kind == "image":
            return True
        return False  # emoji, links, files, video, quoted replies to someone else, …

    def _judge(self, chat, live, history, now):
        """Nobody @-ed the bot: speak up only if (1) some new message isn't noise or aimed at someone else
        (free), (2) the judge model says so, and (3) the cooldowns allow it (a message aimed at the bot
        skips them). Returns the verdict or None."""
        g = self.g
        if not any(self._eligible(ev) for ev in live):
            return None
        t = time.monotonic()
        try:
            images = [h["image_path"] for h in history if h["time"] >= live[0]["time"] and h.get("image_path")]
            d = llm.judge(self.cfg["llm"], g, self._transcript(history, live), images, self._members(chat))
        except Exception as e:
            log(f"[{chat}] judge failed: {type(e).__name__}: {e}")
            return None
        log(f"[{chat}] judge {time.monotonic() - t:.1f}s: to={d['to']} kind={d['kind']} "
            f"reply={d['reply']} — {d['reason'][:80]}")
        why = None
        if not d["reply"]:
            why = "judge: no"
        elif d["to"] != "bot":
            cool, share_cool = (float(g.get("cooldown_minutes", 0)) * 60,
                                float(g.get("share_cooldown_minutes", 0)) * 60)
            recent = self.store.auto_replies(chat, now - max(cool, share_cool))
            last = [t for t, k in recent if k != "bot"]
            shares = [t for t, k in recent if k == "share"]
            if last and now - last[-1] < cool:
                why = f"cooldown: spoke up unasked {(now - last[-1]) / 60:.0f} min ago"
            elif d["kind"] == "share" and shares and now - shares[-1] < share_cool:
                why = f"share cooldown: {(now - shares[-1]) / 60:.0f} min ago"
        self._record(chat, live, d, why)
        if why and d["reply"]:
            log(f"[{chat}] unprompted: {why}")
        return None if why else d

    def _transcript(self, history, live):
        """Plain text for the judge: the last `judge_context` lines, earlier ones and the new burst apart."""
        me = (self.g.get("names") or ["机器人"])[0]
        start = live[0]["time"]
        rows = history[-int(self.g.get("judge_context", 30)):]
        def line(h):
            who = f"{me}（就是你）：" if h["role"] == "assistant" else ""
            return f"[{time.strftime('%H:%M', time.localtime(h['time']))}] {who}{h['text']}"
        old = [line(h) for h in rows if h["time"] < start]
        new = [line(h) for h in rows if h["time"] >= start]
        return ("之前的群聊：\n" + ("\n".join(old) or "（无）") + "\n\n新消息：\n" + "\n".join(new))

    def _record(self, chat, live, verdict, skipped):
        rec = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "chat": chat,
               "new": [ev.get("_text") for ev in live], "verdict": verdict, "skipped": skipped}
        try:
            with open(DECISIONS, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            log(f"[{chat}] could not record decision: {e}")

    def _command(self, chat, live, fresh):
        """开启 / 关闭: a whole text message in a 1:1 chat, "@<me> 开启" in a group, or the same said in a voice
        ("莫名其妙 开启"; while off, heard by a local model only — see _spot). Persistent (survives a
        restart); the confirmation is a fixed text, not the LLM. Returns True when the batch's last
        command was handled."""
        cmd = None
        off_words = self.r.get("off_words", self.r.get("mute_words", ["关闭"]))
        on_words = self.r.get("on_words", self.r.get("unmute_words", ["开启", "打开"]))
        names = self.g.get("names") or []
        summary_words = self.s.get("words", ["总结", "总结一下"])
        for ev in live:
            if ev.get("type") == "text":
                text = ev.get("text") or ""
                if ev.get("is_group"):
                    if not ev.get("mentions_me"):
                        continue
                    text = MENTION.sub("", text)
                word = text.strip(" 。.!！~～")
            elif ev.get("type") == "voice":  # "莫名其妙 关闭" said out loud; in a 1:1 chat the word alone
                text = ev.get("_heard") or ev.get("_text") or ""
                if not text.startswith("[语音] "):
                    continue
                text = text[len("[语音] "):]
                if ev.get("is_group") and not any(n in text for n in names):
                    continue
                for n in names:
                    text = text.replace(n, " ")
                word = text.strip(FILLER)
            else:
                continue
            if word in off_words:
                cmd = ("off", ev)
            elif word in on_words:
                cmd = ("on", ev)
            elif word in summary_words and ev.get("is_group"):  # 1:1 chats just ask the LLM
                cmd = ("summary", ev)
        if not cmd:
            return False
        kind, ev = cmd
        if kind == "summary":
            stop = os.path.exists(os.path.join(ROOT, self.r.get("stop_file", "data/STOP")))
            log(f"[{chat}] summary asked by {ev.get('sender_name') or ev.get('sender')}")
            if not fresh or stop:
                return True
            if not self._on(chat, ev):
                name = names[0] if names else "我的名字"
                self._deliver(chat, f"现在是关闭状态，语音没有转成文字，没法总结。@我 说「开启」，或者发语音说「{name} 开启」。")
            elif self.store.replies_since(chat, time.time() - 60) < int(self.r.get("max_per_minute", 6)):
                self._summarize(chat, time.time(), manual=True)
            return True
        on, group = kind == "on", ev.get("is_group")
        self.store.set_enabled(chat, on, ev["time"])
        if on:  # the LLM's context starts here: nothing from while it was off; WeChat times are whole
            # seconds, the confirmation below is stored with a later float time
            self.store.set_history_floor(chat, ev["time"] + 0.5)
        log(f"[{chat}] switched {'on' if on else 'off'} by {ev.get('sender_name') or ev.get('sender')}")
        if on:
            ack = ((("好的，已开启：语音会自动转成文字，@我就回复；没 @ 我的时候，只在确实帮得上忙时才插一句。"
                     if self.g.get("auto") else "好的，已开启：语音会自动转成文字，@我就回复。")) if group else
                   "好的，已开启：语音会自动转成文字，每条消息我都会回。发「关闭」可以关掉。")
        else:
            name = names[0] if names else "我的名字"
            ack = (f"好的，已关闭：不再回复，语音只在本机听有没有「开启」。想打开就 @我 说「开启」，或者发语音说「{name} 开启」。"
                   if group else "好的，已关闭：不再回复，语音只在本机听有没有「开启」。发「开启」（文字或语音）再打开。")
        if fresh and not os.path.exists(os.path.join(ROOT, self.r.get("stop_file", "data/STOP"))):
            self._deliver(chat, ack)
        return True

    # ---- sending -----------------------------------------------------------
    def _deliver(self, chat, text, auto=None, **meta):
        if self.dry_run:
            log(f"[{chat}] (dry run) would send {text!r}")
            return
        try:
            r = self.sender.send_text(chat, text)
        except NotReady as e:
            with self.lock:
                self.pending[chat] = (time.time(), text, {"auto": auto, **meta})
            log(f"[{chat}] held: {e}")
            self._notice("微信机器人有待发的回复：请在调试版微信里手动发一条文字")
            return
        except Exception as e:  # ambiguous: never retry, a duplicate is worse than a gap
            log(f"[{chat}] send failed, not retried: {type(e).__name__}: {e}")
            return
        meta = {**({"auto": auto} if auto else {}), **{k: v for k, v in meta.items() if v is not None}}
        self.store.add(chat, "assistant", text, client_msg_id=r["clientMessageId"], task_id=r.get("taskId"), **meta)
        log(f"[{chat}] -> sent (task {r.get('taskId')}, result {r.get('result')})")

    def _flush_pending(self):
        ttl = float(self.r.get("pending_ttl_seconds", 300))
        with self.lock:
            items, self.pending = self.pending, {}
        for chat, (created, text, meta) in items.items():
            if time.time() - created > ttl:
                log(f"[{chat}] dropped a held reply ({time.time() - created:.0f}s old)")
                continue
            self._deliver(chat, text, **meta)

    def _notice(self, message):
        if time.time() - self.last_notice < 600:
            return
        self.last_notice = time.time()
        script = "on run argv\ndisplay notification (item 1 of argv) with title \"WeChat bot\"\nend run"
        subprocess.run(["osascript", "-e", script, message], check=False, capture_output=True)
