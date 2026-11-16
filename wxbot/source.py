"""Real-time WeChat (macOS 4.x) message source.

Watches the encrypted message databases with kqueue (no polling), decrypts only the changed
pages in memory, and hands every new message in the watched chats to a callback as a dict.
Read positions are persisted in the store, so a restart resumes where it stopped.
"""
import datetime
import glob
import hashlib
import json
import os
import queue
import re
import select
import threading
import time
import wave
import xml.etree.ElementTree as ET

import zstandard

from .livedb import LiveDB
from .log import log
from .paths import DATA, ROOT
from .wxcrypto import decrypt_image, image_kind, key_for_db

DOCS = os.path.expanduser("~/Library/Containers/com.tencent.xinWeChat/Data/Documents")
TYPES = {1: "text", 3: "image", 34: "voice", 42: "card", 43: "video", 47: "emoji", 48: "location",
         49: "app", 50: "call", 10000: "system", 10002: "system"}
ZSTD = zstandard.ZstdDecompressor()


def account_dir():
    accs = glob.glob(DOCS + "/xwechat_files/wxid_*/")
    if not accs:
        raise SystemExit(f"no WeChat account under {DOCS} (missing permission to read WeChat's container?)")
    return max(accs, key=lambda p: os.stat(p + "db_storage").st_mtime)


def load_keys():
    with open(os.path.join(ROOT, "secrets", "wechat_keys.json")) as f:
        return json.load(f)


def text_of(content, ct_flag):
    if content is None:
        return ""
    raw = content if isinstance(content, (bytes, bytearray)) else content.encode()
    if ct_flag == 4 or raw[:4] == b"\x28\xb5\x2f\xfd":
        raw = ZSTD.decompress(raw, max_output_size=16 << 20)
    return raw.decode("utf-8", "replace")


class Contacts:
    def __init__(self, acc, keys):
        p = acc + "db_storage/contact/contact.db"
        self.db = LiveDB(p, key_for_db(p, keys, acc))
        self.by_user = {}
        self.loaded = 0

    def reload(self, force=False):
        if not force and time.time() - self.loaded < 300:
            return
        self.db.refresh()
        con = self.db.connect()
        self.by_user = {u: dict(remark=r or "", nick=n or "", alias=a or "")
                        for u, r, n, a in con.execute("select username, remark, nick_name, alias from contact")}
        self.loaded = time.time()

    def name(self, user):
        c = self.by_user.get(user)
        return (c and (c["remark"] or c["nick"] or c["alias"])) or user

    def resolve(self, wanted):
        out = set()
        for w in wanted:
            hits = [u for u, c in self.by_user.items() if w in (u, c["remark"], c["nick"], c["alias"])]
            if not hits:
                log(f"warning: chat '{w}' not found in contacts; using it as a username")
                hits = [w]
            out.update(hits)
        return out


class Source:
    """kqueue-driven reader. `on_message(ev)` runs on one worker thread, in arrival order."""

    def __init__(self, cfg, store, on_message, on_wake=lambda: None):
        self.cfg, self.store, self.on_message, self.on_wake = cfg, store, on_message, on_wake
        self.acc = account_dir()
        self.keys = load_keys()
        self.me = re.sub(r"_[0-9a-f]{4}$", "", os.path.basename(self.acc.rstrip("/")))
        self.contacts = Contacts(self.acc, self.keys)
        self.contacts.reload(force=True)
        chats, mention, listen = (cfg.get(k) or [] for k in ("chats", "mention", "listen"))
        if not (chats or mention or listen):
            raise SystemExit("config.toml: `chats`, `mention` and `listen` are empty; list the chats to answer or record")
        # reply mode per chat: always / only when this account is @-mentioned / never (record only)
        self.mode = {u: "never" for u in self.contacts.resolve(listen)}
        self.mode.update({u: "mention" for u in self.contacts.resolve(mention)})
        self.mode.update({u: "always" for u in self.contacts.resolve(chats)})
        self.targets = set(self.mode)
        self.msgdbs, self.last = {}, {}
        self.media = None
        self.events = queue.Queue()
        img = self.keys.get("image", {})
        self.img_key = img.get("aes_key", "").encode()
        self.img_xor = img.get("xor_key") or 0
        os.makedirs(os.path.join(DATA, "media"), mode=0o700, exist_ok=True)
        threading.Thread(target=self._worker, name="source-worker", daemon=True).start()
        log("watching:", ", ".join(f"{self.contacts.name(u)} ({u}) [{self.mode[u]}]" for u in sorted(self.targets)))

    # ---- detection -------------------------------------------------------
    def message_dbs(self):
        return sorted(glob.glob(self.acc + "db_storage/message/message_[0-9]*.db"))

    def scan(self, initial=False):
        for path in self.message_dbs():
            db = self.msgdbs.get(path)
            if db is None:
                db = self.msgdbs[path] = LiveDB(path, key_for_db(path, self.keys, self.acc))
            try:
                changed = db.refresh()
            except Exception as e:  # torn read during a checkpoint: next event retries
                log(f"refresh {os.path.basename(path)} failed: {e}")
                continue
            if not changed and not initial:
                continue
            try:
                self._scan_db(path, db.connect(), initial)
            except Exception as e:
                log(f"query {os.path.basename(path)} failed: {e}")

    def _scan_db(self, path, con, initial):
        id2user = dict(con.execute("select rowid, user_name from Name2Id"))
        tables = {r[0] for r in con.execute("select name from sqlite_master where type='table' and name like 'Msg_%'")}
        for user in self.targets:
            t = "Msg_" + hashlib.md5(user.encode()).hexdigest()
            if t not in tables:
                continue
            k = (path, t)
            if initial:
                saved = self.store.cursor(self._cursor_key(k))
                if saved is None:  # first run for this chat: start at the newest message
                    row = con.execute(f"select max(local_id) from '{t}'").fetchone()
                    saved = row[0] or 0
                    self.store.set_cursor(self._cursor_key(k), saved)
                self.last[k] = saved
            rows = con.execute(
                f"select local_id, server_id, local_type, real_sender_id, create_time, message_content, "
                f"WCDB_CT_message_content, packed_info_data, source, WCDB_CT_source from '{t}' "
                f"where local_id > ? order by local_id",
                (self.last.get(k, 0),)).fetchall()
            if rows:
                try:  # when WeChat wrote this transaction, to time our own pipeline
                    written = os.stat(path + "-wal").st_mtime
                except FileNotFoundError:
                    written = os.stat(path).st_mtime
            for r in rows:
                self.last[k] = r[0]
                self.events.put((k, user, id2user, r, written))

    @staticmethod
    def _cursor_key(k):
        return f"{os.path.basename(k[0])}/{k[1]}"

    # ---- event building (worker thread, keeps order) ----------------------
    def _worker(self):
        while True:
            k, user, id2user, r, written = self.events.get()
            try:
                ev = self.build(user, id2user, r)
                now = time.time()
                ev["latency_ms"] = int((now - ev["time"]) * 1000)  # vs sender timestamp (1 s resolution)
                ev["pipeline_ms"] = int((now - written) * 1000)   # WeChat's DB write -> this event
                self._log(ev)
                self.on_message(ev)
            except Exception as e:
                log(f"event failed: {e!r}")
            self.store.set_cursor(self._cursor_key(k), r[0])

    def build(self, chat, id2user, r):
        local_id, server_id, ltype, sender_id, ctime, content, ct_flag, packed, src, src_flag = r
        base, sub = ltype & 0xFFFFFFFF, ltype >> 32
        text = text_of(content, ct_flag)
        sender = id2user.get(sender_id, "")
        if chat.endswith("@chatroom"):
            m = re.match(r"^([\w\-.@]+):\n", text)
            if m:
                sender, text = m.group(1), text[m.end():]
        ev = dict(chat=chat, chat_name=self.contacts.name(chat), is_group=chat.endswith("@chatroom"),
                  sender=sender, sender_name=self.contacts.name(sender), is_self=sender == self.me,
                  time=ctime, type=TYPES.get(base, str(base)), local_id=local_id, server_id=server_id,
                  reply=self.mode.get(chat, "never"))
        if base == 1 or base >= 10000:
            ev["text"] = text
            if ev["is_group"] and base == 1:
                ev["mentions_me"] = self.mentions_me(text, text_of(src, src_flag))
        elif base == 3:
            ev.update(self.image(chat, ctime, local_id, packed))
        elif base == 34:
            ev.update(self.voice(chat, server_id, local_id, text))
        elif base == 49:
            ev.update(self.appmsg(text, sub))
        elif base == 47:
            m = re.search(r'cdnurl\s*=\s*"([^"]+)"', text)
            ev["emoji_url"] = m and m.group(1).replace("&amp;", "&")
        else:
            ev["xml"] = text
        return ev

    def mentions_me(self, text, source_xml):
        """@ this account: its wxid in the message's <atuserlist> (not @所有人), or "@<nickname>" in the text."""
        m = re.search(r"<atuserlist>(?:<!\[CDATA\[)?([^<\]]*)", source_xml or "")
        if m and self.me in {u.strip() for u in m.group(1).split(",")}:
            return True
        nick = (self.contacts.by_user.get(self.me) or {}).get("nick")
        return bool(nick) and f"@{nick}" in text

    def image(self, chat, ctime, local_id, packed):
        m = re.search(rb"[0-9a-f]{32}", packed or b"")
        if not m:
            return {"image_error": "no file id in message"}
        md5 = m.group(0).decode()
        month = datetime.datetime.fromtimestamp(ctime).strftime("%Y-%m")
        d = os.path.join(self.acc, "msg", "attach", hashlib.md5(chat.encode()).hexdigest(), month, "Img")
        # Bounded wait for WeChat to finish downloading the full-size file (not a resident poll).
        wait_until = time.time() + float(self.cfg.get("media", {}).get("image_wait_seconds", 3))
        while True:
            full = [p for p in (f"{d}/{md5}.dat", f"{d}/{md5}_h.dat") if os.path.exists(p)]
            if full or time.time() >= wait_until:
                break
            time.sleep(0.2)
        cands = full or [p for p in (f"{d}/{md5}_t.dat",) if os.path.exists(p)]
        if not cands:
            return {"image_error": "file not downloaded yet", "image_id": md5}
        img = decrypt_image(open(cands[0], "rb").read(), self.img_key, self.img_xor)
        kind = image_kind(img) or "bin"
        out = os.path.join(DATA, "media", f"{local_id}_{md5}.{kind}")
        with open(out, "wb") as f:
            f.write(img)
        return {"image_path": out, "image_is_thumbnail": not full, "image_format": kind}

    def voice(self, chat, server_id, local_id, xml):
        dur = re.search(r'voicelength="(\d+)"', xml)
        res = {"voice_ms": int(dur.group(1)) if dur else None}
        if self.media is None:
            p = self.acc + "db_storage/message/media_0.db"
            self.media = LiveDB(p, key_for_db(p, self.keys, self.acc))
        data = None
        for _ in range(20):  # the voice blob can land a moment after the message row (≤5 s, bounded)
            self.media.refresh()
            row = self.media.connect().execute(
                "select voice_data from VoiceInfo where svr_id = ? or (svr_id = 0 and local_id = ?) limit 1",
                (server_id, local_id)).fetchone()
            if row:
                data = row[0]
                break
            time.sleep(0.25)
        if not data:
            return {**res, "voice_error": "voice data not found"}
        import pilk
        # local_id is per chat: tag the file with the chat so two chats never overwrite each other
        base = os.path.join(DATA, "media", f"{local_id}_{hashlib.md5(chat.encode()).hexdigest()[:8]}_voice")
        silk = data[1:] if data[:1] == b"\x02" else data
        with open(base + ".silk", "wb") as f:
            f.write(silk)
        pilk.decode(base + ".silk", base + ".pcm", pcm_rate=24000)
        with open(base + ".pcm", "rb") as f:
            pcm = f.read()
        with wave.open(base + ".wav", "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000), w.writeframes(pcm)
        os.remove(base + ".silk"), os.remove(base + ".pcm")
        return {**res, "voice_path": base + ".wav"}

    def appmsg(self, xml, sub):
        res = {"app_type": sub}
        try:
            node = ET.fromstring(xml).find("appmsg")
            if node is not None:
                res.update(title=node.findtext("title"), description=node.findtext("des"),
                           url=node.findtext("url"), app_type=int(node.findtext("type") or sub))
                ref = node.find("refermsg")
                if ref is not None:  # quoted reply
                    res["quote"] = dict(sender=ref.findtext("displayname"), text=ref.findtext("content"))
                    res["text"] = res.pop("title")
        except ET.ParseError:
            res["xml"] = xml
        return res

    def _log(self, ev):
        line = json.dumps(ev, ensure_ascii=False)
        with open(os.path.join(DATA, "messages.jsonl"), "a") as f:
            f.write(line + "\n")

    # ---- watch loop --------------------------------------------------------
    def run(self):
        msg_dir = self.acc + "db_storage/message"
        kq = select.kqueue()
        watched = {}
        fflags = (select.KQ_NOTE_WRITE | select.KQ_NOTE_EXTEND | select.KQ_NOTE_DELETE
                  | select.KQ_NOTE_RENAME | select.KQ_NOTE_ATTRIB)

        def sync():
            want = [msg_dir] + [p + s for p in self.message_dbs() for s in ("", "-wal")]
            for p in want:
                try:
                    ino = os.stat(p).st_ino
                except FileNotFoundError:
                    continue
                if p in watched and watched[p][1] == ino:
                    continue
                if p in watched:
                    os.close(watched[p][0])
                fd = os.open(p, getattr(os, "O_EVTONLY", 0x8000))
                kq.control([select.kevent(fd, filter=select.KQ_FILTER_VNODE,
                                          flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR, fflags=fflags)], 0)
                watched[p] = (fd, ino)

        sync()
        self.scan(initial=True)  # also delivers anything that arrived while we were down
        log(f"ready: {len(self.msgdbs)} message db(s), {len(self.last)} watched chat table(s)")
        while True:
            kq.control(None, 16, None)  # block until WeChat writes; no timeout, no polling
            time.sleep(0.02)            # coalesce the burst of writes from one transaction
            kq.control(None, 64, 0)
            sync()
            self.contacts.reload()
            self.scan()
            self.on_wake()


def list_chats(q):
    acc, keys = account_dir(), load_keys()
    contacts = Contacts(acc, keys)
    contacts.reload(force=True)
    p = acc + "db_storage/session/session.db"
    s = LiveDB(p, key_for_db(p, keys, acc))
    s.refresh()
    rows = s.connect().execute("select username, sort_timestamp from SessionTable order by sort_timestamp desc")
    for user, ts in rows:
        name = contacts.name(user)
        if q and q not in name and q not in user:
            continue
        when = datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts else ""
        print(f"{when:12} {name}  [{user}]")
