"""Unit tests for wxbot (crypto, store, LLM policy, pipeline, source, native sender)."""
import hashlib
import os
import queue
import sqlite3
import struct
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from wxbot import llm, wxcrypto
from wxbot.pipeline import Pipeline
from wxbot.sender.native import NativeSender, _image_complete
from wxbot.source import Source
from wxbot.store import Store


class TestWxCrypto(unittest.TestCase):
    def test_v2_image_decrypt_and_pkcs7_validation(self):
        key = b"0123456789abcdef"
        plain_head = b"\xff\xd8\xff\xe0hello"
        pad = 16 - len(plain_head)
        enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
        aes_part = enc.update(plain_head + bytes([pad]) * pad) + enc.finalize()
        mid, xor_plain, xor_key = b"MID", b"TAIL", 0x5A
        v2 = (wxcrypto.V2_MAGIC + struct.pack("<II", len(plain_head), len(xor_plain)) + b"\x00"
              + aes_part + mid + bytes(b ^ xor_key for b in xor_plain))
        self.assertEqual(wxcrypto.decrypt_image(v2, key, xor_key), plain_head + mid + xor_plain)

        with self.assertRaisesRegex(ValueError, "aes_key"):
            wxcrypto.decrypt_image(v2, b"", xor_key)

        bad_enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
        bad_aes = bad_enc.update(b"0123456789abcde\x00") + bad_enc.finalize()
        bad_v2 = wxcrypto.V2_MAGIC + struct.pack("<II", 15, 0) + b"\x00" + bad_aes
        with self.assertRaisesRegex(ValueError, "PKCS7"):
            wxcrypto.decrypt_image(bad_v2, key, xor_key)

    def test_wal_pages_committed_only_and_salt_stop(self):
        with tempfile.TemporaryDirectory() as d:
            wal_p = os.path.join(d, "test.db-wal")
            salt = b"\x01\x02\x03\x04\x05\x06\x07\x08"
            old_salt = b"\xff" * 8
            header = struct.pack(">IIII", 0x377F0682, 3007000, wxcrypto.PAGE, 1) + salt + b"\x00" * 8
            frame1 = struct.pack(">II", 1, 0) + salt + b"\x00" * 8 + b"A" * wxcrypto.PAGE
            frame2 = struct.pack(">II", 2, 2) + salt + b"\x00" * 8 + b"B" * wxcrypto.PAGE
            uncommitted = struct.pack(">II", 3, 0) + salt + b"\x00" * 8 + b"C" * wxcrypto.PAGE
            stale = struct.pack(">II", 1, 5) + old_salt + b"\x00" * 8 + b"X" * wxcrypto.PAGE
            with open(wal_p, "wb") as f:
                f.write(header + frame1 + frame2 + uncommitted + stale)
            pages, db_size = wxcrypto.wal_pages(wal_p)
            self.assertEqual(db_size, 2)
            self.assertEqual(set(pages), {1, 2})
            self.assertEqual(pages[1], b"A" * wxcrypto.PAGE)


class TestStore(unittest.TestCase):
    def test_is_on_history_order_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bot.db")
            st = Store(p)
            try:
                self.assertTrue(st.is_on("c1", "always"))
                self.assertTrue(st.is_on("c1", "mention"))
                self.assertFalse(st.is_on("c1", "never"))
                st.set_enabled("c1", False, 100.0)
                self.assertFalse(st.is_on("c1", "always"))
                st.set_enabled("c2", True, 100.0)
                self.assertTrue(st.is_on("c2", "never"))

                # Causal ordering by id even if assistant clock > next user's server timestamp
                id1 = st.add("c1", "user", "u1", at=1000)
                id2 = st.add("c1", "assistant", "a1", at=1010)
                id3 = st.add("c1", "user", "u2", at=1003)
                hist = st.history("c1", 900)
                self.assertEqual([h["id"] for h in hist], [id1, id2, id3])

                # Checkpoint truncates WAL while leaving the connection usable
                st.checkpoint()
                wal = p + "-wal"
                self.assertTrue(not os.path.exists(wal) or os.path.getsize(wal) == 0)
                st.add("c1", "assistant", "after checkpoint", at=1020)
                self.assertEqual(len(st.history("c1", 900)), 4)
            finally:
                st.db.close()


class TestLLM(unittest.TestCase):
    def test_clean_and_missing_image(self):
        self.assertEqual(llm._clean("今天晴 ([天气网](https://example.com))，**注意保暖**"), "今天晴，注意保暖")
        self.assertEqual(llm._clean("详见[公告](https://example.com/a)"), "详见公告")
        self.assertIsNone(llm._image_part("/nonexistent/wxbot_deleted.jpg"))
        inp = llm.build_input([{"role": "user", "text": "[图片]", "image_path": "/nonexistent/deleted.jpg"}], 3)
        self.assertEqual(len(inp[0]["content"]), 1)

    def test_decisions_policy_thresholds(self):
        def mock_post(to, kind, facts):
            answers = [{"name": "to", "type": "choice", "choice": to, "confidence": 0.95},
                       {"name": "kind", "type": "choice", "choice": kind, "confidence": 0.95}]
            answers += [{"name": k, "type": "predicate", "probability": v} for k, v in facts.items()]
            return SimpleNamespace(status_code=200, json=lambda: {"answers": answers}, text="")

        gcfg = {"judge_api": "decisions", "judge_prompt": "名字{names}", "names": ["助手"],
                "reply_threshold": 0.75, "fact_thresholds": {"answerable": 0.4}, "voice_reply_threshold": 0.97}
        lcfg = {"model": "gpt-test", "timeout_seconds": 10}
        zeros = {k: 0.0 for k in llm.FACTS}

        # to=member always stays quiet even with high facts
        with patch("wxbot.llm.httpx.post", return_value=mock_post("member", "question", {**zeros, "answerable": 0.99, "unanswered": 0.99})), \
             patch("wxbot.llm.keys.get", return_value="sk-test"):
            d = llm.judge(lcfg, gcfg, "transcript")
            self.assertFalse(d["reply"])

        # question with per-fact answerable=0.45 (>= 0.4) and unanswered=0.80 (>= 0.75) replies on text...
        qfacts = {**zeros, "answerable": 0.45, "unanswered": 0.80}
        with patch("wxbot.llm.httpx.post", return_value=mock_post("group", "question", qfacts)), \
             patch("wxbot.llm.keys.get", return_value="sk-test"):
            d_txt = llm.judge(lcfg, gcfg, "transcript", voice=False)
            d_voice = llm.judge(lcfg, gcfg, "transcript", voice=True)
            self.assertTrue(d_txt["reply"] and d_txt["rule"] == "question")
            self.assertFalse(d_voice["reply"])  # voice requires >= 0.97

        # chat with hot_topic=0.80 triggers topic rule
        with patch("wxbot.llm.httpx.post", return_value=mock_post("group", "chat", {**zeros, "hot_topic": 0.80})), \
             patch("wxbot.llm.keys.get", return_value="sk-test"):
            d = llm.judge(lcfg, gcfg, "transcript")
            self.assertTrue(d["reply"] and d["rule"] == "topic")


class TestPipeline(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.stop_file = os.path.join(tmp.name, "STOP")  # not the live bot's data/STOP

    def _make_pipe(self, st, sent, **group_overrides):
        cfg = {
            "reply": {"debounce_seconds": 0.08, "history_hours": 12, "max_age_seconds": 600, "max_per_minute": 6,
                      "stop_file": self.stop_file},
            "group": {"auto": True, "names": ["小助"], "wait_seconds": 0.15, "judge_context": 30, **group_overrides},
            "summary": {"auto": False},
            "llm": {"model": "gpt-test", "system_prompt": "sys"},
            "stt": {"provider": "combo"},
        }
        sender = MagicMock()
        sender.send_text.side_effect = lambda to, text: (
            sent.append((to, text)) or {"clientMessageId": len(sent), "taskId": len(sent), "result": 0}
        )
        return Pipeline(cfg, st, sender, dry_run=False)

    def test_transcript_partition_and_command_burst(self):
        with tempfile.TemporaryDirectory() as d:
            st = Store(os.path.join(d, "bot.db"))
            try:
                sent = []
                pipe = self._make_pipe(st, sent)
                now = int(time.time())
                chat = "g1@chatroom"
                st.add(chat, "user", "甲：旧消息", at=now - 10, local_id=1, type="text")
                st.add(chat, "assistant", "旧回复", at=now, client_msg_id=1)  # clock ahead of next msg
                ev2 = {"chat": chat, "local_id": 2, "time": now - 5, "type": "text", "text": "新消息",
                       "is_group": True, "sender": "u2", "sender_name": "乙", "reply": "mention", "is_self": False}
                pipe._ingest(chat, ev2)
                hist = st.history(chat, now - 3600)
                old_part, new_part = pipe._transcript(hist, [ev2]).split("\n\n新消息：\n")
                self.assertIn("旧回复", old_part)
                self.assertNotIn("旧回复", new_part)
                self.assertIn("新消息", new_part)

                # 开启 followed immediately in the same integer second by a question must not be swallowed
                chat2 = "wxid_u1"
                ev_on = {"chat": chat2, "local_id": 10, "time": now, "type": "text", "text": "开启",
                         "is_group": False, "sender": chat2, "sender_name": "友", "reply": "never", "is_self": False}
                ev_q = {"chat": chat2, "local_id": 11, "time": now, "type": "text", "text": "今天天气",
                        "is_group": False, "sender": chat2, "sender_name": "友", "reply": "never", "is_self": False}
                seen_hist = []
                with patch("wxbot.pipeline.llm.reply", side_effect=lambda c, h, **kw: (seen_hist.append(h) or "晴")):
                    pipe.on_message(ev_on)
                    pipe.on_message(ev_q)
                    end = time.time() + 3.0
                    while len(sent) < 2 and time.time() < end:
                        time.sleep(0.03)
                self.assertEqual(len(sent), 2)
                self.assertIn("已开启", sent[0][1])
                self.assertEqual(sent[1][1], "晴")
                self.assertTrue(any("今天天气" in h["text"] for h in seen_hist[0]))
            finally:
                st.db.close()

    def test_share_cooldown_checks_auto_kind_not_raw_kind(self):
        with tempfile.TemporaryDirectory() as d:
            st = Store(os.path.join(d, "bot.db"))
            try:
                pipe = self._make_pipe(st, [], share_cooldown_minutes=30, cooldown_minutes=0)
                now = time.time()
                chat = "g1@chatroom"
                st.add(chat, "assistant", "好看", at=now - 60, auto="share")
                ev = {"chat": chat, "local_id": 5, "time": now, "type": "image", "_text": "[图片]",
                      "_row_id": 99, "is_group": True}
                # A share triggered by the "topic" rule (e.g. stock screenshot) must NOT be blocked by share_cooldown
                verdict = {"to": "group", "kind": "share", "rule": "topic", "reply": True, "reason": "hot_topic"}
                with patch("wxbot.pipeline.llm.judge", return_value=verdict), \
                     patch.object(pipe, "_record"):
                    self.assertIsNotNone(pipe._judge(chat, [ev], [], now))
                # Whereas a share triggered by the "share" rule IS blocked by share_cooldown
                verdict_share = {"to": "group", "kind": "share", "rule": "share", "reply": True, "reason": "share"}
                with patch("wxbot.pipeline.llm.judge", return_value=verdict_share), \
                     patch.object(pipe, "_record"):
                    self.assertIsNone(pipe._judge(chat, [ev], [], now))
            finally:
                st.db.close()


class TestSource(unittest.TestCase):
    def test_should_fetch_cursors_and_shard_sort(self):
        self.assertEqual(
            sorted(["/a/media_2.db", "/a/media_10.db", "/a/media_0.db"], key=Source._shard_no, reverse=True),
            ["/a/media_10.db", "/a/media_2.db", "/a/media_0.db"],
        )
        with tempfile.TemporaryDirectory() as d:
            st = Store(os.path.join(d, "bot.db"))
            try:
                src = object.__new__(Source)
                src.cfg = {"media": {"fetch_full": True}, "reply": {"max_age_seconds": 600}}
                src.store = st
                src.started = time.time()
                src.fetch_image = lambda *a: b""
                src.mode = {"listen_c": "never", "mention_c": "mention"}
                now = time.time()

                self.assertFalse(src.should_fetch("listen_c", now))
                self.assertTrue(src.should_fetch("mention_c", now))
                self.assertFalse(src.should_fetch("mention_c", now, is_self=True))
                st.set_enabled("listen_c", True, now)
                st.set_enabled("mention_c", False, now)
                self.assertTrue(src.should_fetch("listen_c", now))
                self.assertFalse(src.should_fetch("mention_c", now))

                # Cold-start torn read recovers without replaying old history
                chat = "wxid_test"
                tbl = "Msg_" + hashlib.md5(chat.encode()).hexdigest()
                db_path = os.path.join(d, "message_0.db")
                open(db_path, "wb").close()

                class FakeDB:
                    fail_refresh = True
                    def refresh(self):
                        if self.fail_refresh:
                            raise RuntimeError("torn")
                        return False
                    def reset(self):
                        pass
                    def connect(self):
                        con = sqlite3.connect(":memory:")
                        con.execute("create table Name2Id (user_name text)")
                        con.execute("insert into Name2Id values (?)", (chat,))
                        con.execute(f"create table '{tbl}' (local_id int, server_id int, local_type int, "
                                    f"real_sender_id int, create_time int, message_content text, "
                                    f"WCDB_CT_message_content int, packed_info_data blob, source text, WCDB_CT_source int)")
                        con.execute(f"insert into '{tbl}' values (1, 1, 1, 1, ?, 'old', 0, ?, '', 0)", (int(now), b""))
                        return con

                fdb = FakeDB()
                src.targets, src.msgdbs, src.last = {chat}, {db_path: fdb}, {}
                src._pending_init, src._retry_dbs = {db_path}, set()
                src.events = queue.Queue()
                src.message_dbs = lambda: [db_path]
                src.acc, src.keys = d, {}
                st.set_cursor(src._cursor_key((db_path, tbl)), 1)

                src.scan(initial=True)
                self.assertIn(db_path, src._pending_init)
                fdb.fail_refresh = False
                src.scan(initial=False)
                self.assertNotIn(db_path, src._pending_init)
                self.assertTrue(src.events.empty())
            finally:
                st.db.close()

    def test_quote_chatusr_and_decrypt_empty(self):
        src = object.__new__(Source)
        src.me = "wxid_bot"
        src.cfg = {"group": {"names": ["小助"]}}
        src.contacts = SimpleNamespace(by_user={"wxid_bot": {"nick": "小助"}}, name=lambda u: u)
        src.mode = {"g@chatroom": "mention"}
        src.img_key, src.img_xor = b"0123456789abcdef", 0x5A

        xml_me = "<msg><appmsg><title>hi</title><type>57</type><refermsg><type>1</type><chatusr>wxid_bot</chatusr><displayname>别称</displayname><content>x</content></refermsg></appmsg></msg>"
        xml_other = "<msg><appmsg><title>hi</title><type>57</type><refermsg><type>1</type><chatusr>wxid_other</chatusr><displayname>小助</displayname><content>x</content></refermsg></appmsg></msg>"
        self.assertTrue(src.build("g@chatroom", {1: "u"}, (1, 1, 49, 1, 1000, xml_me, 0, b"", "", 0))["quotes_me"])
        self.assertFalse(src.build("g@chatroom", {1: "u"}, (2, 2, 49, 1, 1000, xml_other, 0, b"", "", 0))["quotes_me"])

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "empty.dat")
            open(p, "wb").close()
            with patch("wxbot.source.time.sleep", lambda _: None):
                self.assertIsNone(src._decrypt_file(p))


class TestNativeSender(unittest.TestCase):
    def test_close_no_deadlock_and_eternalize(self):
        sender = NativeSender()
        eternalized = []

        class FakeScript:
            exports_sync = SimpleNamespace(status=lambda: {"inFlight": 0})
            def eternalize(self):
                eternalized.append(True)

        class FakeSession:
            def detach(self):
                sender._on_detached(123, "application-requested")

        sender.session, sender.script, sender.pid = FakeSession(), FakeScript(), 123
        done = threading.Event()
        threading.Thread(target=lambda: (sender.close(), done.set()), daemon=True).start()
        self.assertTrue(done.wait(2.0), "NativeSender.close() deadlocked")
        self.assertEqual(eternalized, [])

        sender.session, sender.script, sender.pid = FakeSession(), FakeScript(), 123
        sender._active_downloads = 1
        sender.close()
        self.assertEqual(eternalized, [True])

    def test_attach_failure_detaches(self):
        sender = NativeSender()
        detached = []
        fake_script = SimpleNamespace(on=lambda *a: None, load=lambda: None,
                                      exports_sync=SimpleNamespace(initialize=MagicMock(side_effect=RuntimeError("not ready"))))
        fake_session = SimpleNamespace(create_script=lambda _: fake_script, on=lambda *a: None,
                                       detach=lambda: detached.append(True))
        with patch("wxbot.sender.native.debug_pid", return_value=99), \
             patch("wxbot.sender.native.frida.attach", return_value=fake_session):
            with self.assertRaises(RuntimeError):
                sender.ensure()
        self.assertEqual(detached, [True])

    def test_image_complete(self):
        self.assertTrue(_image_complete(b"\xff\xd8\xff\xe0x\xff\xd9"))
        self.assertFalse(_image_complete(b"\xff\xd8\xff\xe0trunc"))
        self.assertTrue(_image_complete(b"\x89PNG\r\n\x1a\nIEND\xaeB`\x82"))
        self.assertFalse(_image_complete(b"\x89PNG\r\n\x1a\ntrunc"))
        self.assertTrue(_image_complete(b"GIF89a\x3b"))
        body = b"WEBP" + b"0" * 16
        webp = b"RIFF" + struct.pack("<I", len(body)) + body
        self.assertTrue(_image_complete(webp))
        self.assertFalse(_image_complete(webp[:-4]))
        self.assertIsNone(_image_complete(b"wxgf\x10\x00\x00\x00data"))
