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
from .sender import Refused
from .wxcrypto import MissingImageKey, decrypt_image, image_kind, key_for_db, wxgf_to_jpeg

DOCS = os.path.expanduser("~/Library/Containers/com.tencent.xinWeChat/Data/Documents")
TYPES = {1: "text", 3: "image", 34: "voice", 42: "card", 43: "video", 47: "emoji", 48: "location",
         49: "app", 50: "call", 10000: "system", 10002: "system"}
ZSTD = zstandard.ZstdDecompressor()
QUOTED = {"3": "[图片]", "34": "[语音]", "43": "[视频]", "47": "[表情]", "48": "[位置]", "42": "[名片]"}
BUILD_ATTEMPTS = 3


def account_dir():
    accs = glob.glob(DOCS + "/xwechat_files/wxid_*/")
    if not accs:
        raise SystemExit(f"no WeChat account under {DOCS} (missing permission to read WeChat's container?)")
    return max(accs, key=lambda p: os.stat(p + "db_storage").st_mtime)


def load_keys():
    with open(os.path.join(ROOT, "secrets", "wechat_keys.json")) as f:
        return json.load(f)


def wait_for(path, seconds):
    """Bounded wait for a file WeChat is about to write (not a resident poll)."""
    end = time.time() + seconds
    while not os.path.exists(path) and time.time() < end:
        time.sleep(0.1)
    return os.path.exists(path)


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
        try:
            self.by_user = {u: dict(remark=r or "", nick=n or "", alias=a or "")
                            for u, r, n, a in con.execute("select username, remark, nick_name, alias from contact")}
        finally:
            con.close()
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

    def __init__(self, cfg, store, on_message, on_wake=lambda: None, fetch_image=None):
        """fetch_image(me, cdn_url, aes_key, timeout) -> bytes | None: full-size image via WeChat."""
        self.cfg, self.store, self.on_message, self.on_wake = cfg, store, on_message, on_wake
        self.fetch_image = fetch_image
        self.started = time.time()
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
        self._pending_init = set(self.message_dbs())
        self._retry_dbs = set()
        self.mediadbs = {}
        self.events = queue.Queue()
        img = self.keys.get("image", {})
        self.img_key = img.get("aes_key", "").encode()
        self.img_xor = img.get("xor_key") or 0
        os.makedirs(os.path.join(DATA, "media"), mode=0o700, exist_ok=True)
        threading.Thread(target=self._worker, name="source-worker", daemon=True).start()
        log("watching:", ", ".join(f"{self.contacts.name(u)} ({u}) [{self.mode[u]}]" for u in sorted(self.targets)))

    # ---- detection -------------------------------------------------------
    @staticmethod
    def _shard_no(path):
        return int(re.search(r"_(\d+)\.db$", path).group(1))

    def message_dbs(self):
        return sorted(glob.glob(self.acc + "db_storage/message/message_[0-9]*.db"), key=self._shard_no)

    def media_dbs(self):
        """Newest shard first: that is where a just-arrived voice lands."""
        return sorted(glob.glob(self.acc + "db_storage/message/media_[0-9]*.db"), key=self._shard_no, reverse=True)

    def scan(self, initial=False):
        for path in self.message_dbs():
            is_init = initial or path in self._pending_init
            must_scan = is_init or path in self._retry_dbs
            db = self.msgdbs.get(path)
            try:
                if db is None:
                    db = self.msgdbs[path] = LiveDB(path, key_for_db(path, self.keys, self.acc))
                changed = db.refresh()
            except Exception as e:  # torn read during a checkpoint: next event retries
                log(f"refresh {os.path.basename(path)} failed: {e}")
                self._retry_dbs.add(path)
                continue
            if not changed and not must_scan:
                continue
            con = None
            try:
                con = db.connect()
                self._scan_db(path, con, is_init)
            except Exception as e:
                log(f"query {os.path.basename(path)} failed: {e}")
                db.reset()
                self._retry_dbs.add(path)
            else:
                self._pending_init.discard(path)
                self._retry_dbs.discard(path)
            finally:
                if con is not None:
                    con.close()

    def _scan_db(self, path, con, is_init):
        id2user = dict(con.execute("select rowid, user_name from Name2Id"))
        tables = {r[0] for r in con.execute("select name from sqlite_master where type='table' and name like 'Msg_%'")}
        for user in self.targets:
            t = "Msg_" + hashlib.md5(user.encode()).hexdigest()
            if t not in tables:
                continue
            k = (path, t)
            if k not in self.last:
                saved = self.store.cursor(self._cursor_key(k))
                if saved is None:
                    newest = con.execute(f"select coalesce(max(local_id), 0) from '{t}'").fetchone()[0]
                    if is_init:  # first run for this chat: start at the newest message
                        saved = newest
                    else:  # table or shard created while running: deliver what arrived since the bot
                        # started, not history WeChat may have copied into it (sync, migration)
                        first = con.execute(f"select min(local_id) from '{t}' where create_time >= ?",
                                            (int(self.started) - 60,)).fetchone()[0]
                        saved = newest if first is None else first - 1
                    self.store.set_cursor(self._cursor_key(k), saved)
                self.last[k] = saved
            rows = con.execute(
                f"select local_id, server_id, local_type, real_sender_id, create_time, message_content, "
                f"WCDB_CT_message_content, packed_info_data, source, WCDB_CT_source from '{t}' "
                f"where local_id > ? order by local_id",
                (self.last[k],)).fetchall()
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
            ev = None
            for attempt in range(BUILD_ATTEMPTS):  # reading voice/image data can hit a torn read: retry here,
                try:                               # since later rows of the chat would move the cursor past it
                    ev = self.build(user, id2user, r)
                    break
                except Exception as e:
                    log(f"event {r[0]} build failed ({attempt + 1}/{BUILD_ATTEMPTS}): {e!r}")
                    if attempt + 1 < BUILD_ATTEMPTS:
                        time.sleep(1.0)
            if ev is None:  # nothing was acted on, so a restart may safely deliver it again
                log(f"event {r[0]} dropped for this run; cursor not advanced")
                continue
            try:
                now = time.time()
                ev["latency_ms"] = int((now - ev["time"]) * 1000)  # vs sender timestamp (1 s resolution)
                ev["pipeline_ms"] = int((now - written) * 1000)   # WeChat's DB write -> this event
                self._log(ev)
                self.on_message(ev)
            except Exception as e:
                log(f"event failed: {e!r}")
            try:  # advance even if on_message raised: it may already have replied, a replay would reply twice
                self.store.set_cursor(self._cursor_key(k), r[0])
            except Exception as e:
                log(f"cursor update failed: {e!r}")

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
                ev["mentions_me"], ev["at_others"] = self.mentions(text, text_of(src, src_flag))
        elif base == 3:
            ev.update(self.image(chat, ctime, local_id, packed, text, is_self=ev["is_self"]))
        elif base == 34:
            ev.update(self.voice(chat, server_id, local_id, text))
        elif base == 49:
            ev.update(self.appmsg(text, sub))
            if ev["is_group"] and ev.get("quote"):  # a quoted reply: to the bot's message, or with an @
                q = ev["quote"]
                qid = q.get("sender_id")
                ev["quotes_me"] = (qid == self.me) if (qid and not qid.endswith("@chatroom")) else (
                    q.get("sender") in self.my_names())
                ev["mentions_me"], ev["at_others"] = self.mentions(ev.get("text") or "", text_of(src, src_flag))
        elif base == 47:
            m = re.search(r'cdnurl\s*=\s*"([^"]+)"', text)
            ev["emoji_url"] = m and m.group(1).replace("&amp;", "&")
        else:
            ev["xml"] = text
        return ev

    def my_names(self):
        """What group members see as this account: its nickname plus `[group] names`."""
        nick = (self.contacts.by_user.get(self.me) or {}).get("nick")
        return {n for n in [nick, *self.cfg.get("group", {}).get("names", [])] if n}

    def mentions(self, text, source_xml):
        """(@ this account, @ someone else). Me: my wxid in the message's <atuserlist>, or "@<my name>" in
        the text. Someone else: any other wxid there, or @所有人 (an announcement, not a question to me)."""
        m = re.search(r"<atuserlist>(?:<!\[CDATA\[)?([^<\]]*)", source_xml or "")
        users = {u.strip() for u in m.group(1).split(",") if u.strip()} if m else set()
        me = self.me in users or any(f"@{n}" in text for n in self.my_names())
        others = bool(users - {self.me, "notify@all"}) or "notify@all" in users or "@所有人" in text
        return me, others

    def image(self, chat, ctime, local_id, packed, xml, is_self=False):
        m = re.search(rb"[0-9a-f]{32}", packed or b"")
        if not m:
            return {"image_error": "no file id in message"}
        md5 = m.group(0).decode()
        media = self.cfg.get("media", {})
        month = datetime.datetime.fromtimestamp(ctime).strftime("%Y-%m")
        d = os.path.join(self.acc, "msg", "attach", hashlib.md5(chat.encode()).hexdigest(), month, "Img")
        # WeChat itself only downloads the full size when the image is opened in the client.
        wait_until = time.time() + float(media.get("image_wait_seconds", 0))
        while True:  # optional bounded wait for that (not a resident poll)
            full = [p for p in (f"{d}/{md5}_h.dat", f"{d}/{md5}.dat") if os.path.exists(p)]
            if full or time.time() >= wait_until:
                break
            time.sleep(0.2)
        img, origin = None, None
        thumb = f"{d}/{md5}_t.dat"
        if full:
            img = self._decrypt_file(full[0])
            origin = img and "local"
        elif self.should_fetch(chat, ctime, is_self=is_self):
            # The message row lands before the client has downloaded the thumbnail, and a request for
            # the same image while that runs is refused (-20003): let the thumbnail arrive first.
            wait_for(thumb, 5)
            img = self.fetch_full(local_id, xml, float(media.get("fetch_timeout_seconds", 10)))
            origin = img and "cdn"
            full = [] if img else [p for p in (f"{d}/{md5}_h.dat", f"{d}/{md5}.dat") if os.path.exists(p)]
            if full:  # meanwhile the client fetched it itself (chat open)
                img = self._decrypt_file(full[0])
                origin = img and "local"
        if img is None:
            if not wait_for(thumb, 3):
                return {"image_error": "file not downloaded yet", "image_id": md5}
            img, origin = self._decrypt_file(thumb), "thumbnail"  # may still be being written
            if img is None:
                return {"image_error": "image could not be decrypted", "image_id": md5}
        if image_kind(img) == "wxgf":  # WeChat's own HEVC container: the LLM needs JPEG
            img = wxgf_to_jpeg(img) or img
        kind = image_kind(img) or "bin"
        out = os.path.join(DATA, "media", f"{local_id}_{md5}.{kind}")
        with open(out, "wb") as f:
            f.write(img)
        return {"image_path": out, "image_is_thumbnail": origin == "thumbnail", "image_origin": origin,
                "image_format": kind}

    def _decrypt_file(self, path, tries=3):
        """Decrypted image bytes, or None (logged) for a missing key or a file WeChat is still writing:
        a bad image must not cost the whole message."""
        for attempt in range(tries):
            try:
                with open(path, "rb") as f:
                    data = f.read()
                if not data:
                    raise ValueError("empty image file")
                return decrypt_image(data, self.img_key, self.img_xor)
            except (OSError, ValueError) as e:
                err = e
                if isinstance(e, MissingImageKey):  # permanent: don't stall the source worker retrying
                    break
                if attempt + 1 < tries:
                    time.sleep(0.3)
        log(f"image {os.path.basename(path)}: {err}")
        return None

    def should_fetch(self, chat, ctime, is_self=False):
        """Fetch full size only for chats currently 开启, and not for own messages or old backlog."""
        return (not is_self and self.fetch_image is not None
                and self.cfg.get("media", {}).get("fetch_full", True)
                and self.store.is_on(chat, self.mode.get(chat))
                and time.time() - ctime <= float(self.cfg.get("reply", {}).get("max_age_seconds", 600)))

    def fetch_full(self, local_id, xml, timeout):
        attr = lambda name: (re.search(rf'\b{name}="([0-9a-fA-F]+)"', xml or "") or [None, None])[1]
        url, key = attr("cdnmidimgurl"), attr("aeskey")
        if not (url and key):
            return None
        t0 = time.monotonic()
        deadline = t0 + timeout
        while True:
            try:
                img = self.fetch_image(self.me, url.lower(), key.lower(), max(1.0, deadline - time.monotonic()))
                break
            except Refused as e:  # usually the client is still fetching this image itself
                if time.monotonic() + 1 >= deadline:
                    log(f"image {local_id}: full size not fetched ({e})")
                    return None
                time.sleep(1)
            except Exception as e:  # WeChat-debug not running, ...
                log(f"image {local_id}: full size not fetched ({type(e).__name__}: {e})")
                return None
        ms = int((time.monotonic() - t0) * 1000)
        if not img or not image_kind(img):
            log(f"image {local_id}: full size not fetched within {timeout:g}s")
            return None
        log(f"image {local_id}: full size fetched ({len(img) // 1024} KB, {ms} ms)")
        return img

    def _voice_blob(self, path, sql, params):
        db = self.mediadbs.get(path)
        con = None
        try:
            if db is None:
                db = self.mediadbs[path] = LiveDB(path, key_for_db(path, self.keys, self.acc))
            db.refresh()
            con = db.connect()
            row = con.execute(sql, params).fetchone()
        except Exception as e:  # torn read during a checkpoint: drop the mirror, the next attempt re-reads
            log(f"voice lookup in {os.path.basename(path)} failed: {e}")
            if db is not None:
                db.reset()
            return None
        finally:
            if con is not None:
                con.close()
        return row[0] if row else None

    def voice(self, chat, server_id, local_id, xml):
        dur = re.search(r'voicelength="(\d+)"', xml)
        res = {"voice_ms": int(dur.group(1)) if dur else None}
        # local_id is per chat: match it only within this chat (media_*.db has its own Name2Id)
        local = "local_id = ? and chat_name_id = (select rowid from Name2Id where user_name = ?)"
        if server_id:
            sql = f"select voice_data from VoiceInfo where svr_id = ? or (svr_id = 0 and {local}) limit 1"
            params = (server_id, local_id, chat)
        else:  # own voice not yet acknowledged by the server
            sql = f"select voice_data from VoiceInfo where {local} limit 1"
            params = (local_id, chat)
        data = None
        shards = self.media_dbs()
        for attempt in range(20):  # the voice blob can land a moment after the message row (≤5 s, bounded)
            # first pass: every shard; retries: only the newest, where a late blob is written
            for p in shards if attempt == 0 else shards[:1]:
                data = self._voice_blob(p, sql, params)
                if data:
                    break
            if data:
                break
            time.sleep(0.25)
        if not data:
            return {**res, "voice_error": "voice data not found"}
        import pilk
        # local_id is per chat: tag the file with the chat so two chats never overwrite each other
        base = os.path.join(DATA, "media", f"{local_id}_{hashlib.md5(chat.encode()).hexdigest()[:8]}_voice")
        silk = data[1:] if data[:1] == b"\x02" else data
        try:
            with open(base + ".silk", "wb") as f:
                f.write(silk)
            pilk.decode(base + ".silk", base + ".pcm", pcm_rate=24000)
            with open(base + ".pcm", "rb") as f:
                pcm = f.read()
            with wave.open(base + ".wav", "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000), w.writeframes(pcm)
        finally:
            for ext in (".silk", ".pcm"):
                try:
                    os.remove(base + ext)
                except FileNotFoundError:
                    pass
        return {**res, "voice_path": base + ".wav"}

    @staticmethod
    def quoted(kind, content):
        """What a quoted message said: its text, or a placeholder instead of an image's / file's XML."""
        content = re.sub(r"^[\w\-.@]+:\n", "", content or "")  # group quotes carry "wxid:\n"
        if kind in (None, "", "1"):
            return content
        if kind == "49":  # a quoted link / file / quoted reply: its title
            try:
                title = ET.fromstring(content).findtext("appmsg/title")
            except ET.ParseError:
                title = None
            return title or "[链接/文件]"
        return QUOTED.get(kind, "[消息]")

    def appmsg(self, xml, sub):
        res = {"app_type": sub}
        try:
            node = ET.fromstring(xml).find("appmsg")
            if node is not None:
                res.update(title=node.findtext("title"), description=node.findtext("des"),
                           url=node.findtext("url"), app_type=int(node.findtext("type") or sub))
                ref = node.find("refermsg")
                if ref is not None:  # quoted reply
                    q = dict(sender=ref.findtext("displayname"),
                             text=self.quoted(ref.findtext("type"), ref.findtext("content")))
                    if (chatusr := (ref.findtext("chatusr") or "").strip()):
                        q["sender_id"] = chatusr
                    res["quote"] = q
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
                    if p in watched:
                        os.close(watched.pop(p)[0])
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
    con = s.connect()
    try:
        rows = con.execute("select username, sort_timestamp from SessionTable order by sort_timestamp desc").fetchall()
    finally:
        con.close()
    for user, ts in rows:
        name = contacts.name(user)
        if q and q not in name and q not in user:
            continue
        when = datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts else ""
        print(f"{when:12} {name}  [{user}]")
