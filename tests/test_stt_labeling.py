"""Unit tests for stt and labeling (CER, refine, whisper chunking, dataset, chat_view, label HTTP)."""
import io
import json
import os
import re
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from labeling import chat_view, dataset, label
from stt import cer, core as stt_core, local_whisper_wu, refine


class TestSTT(unittest.TestCase):
    def test_cer(self):
        self.assertEqual(cer.cer("你好，世界！", "你好世界"), 0.0)
        self.assertAlmostEqual(cer.cer("今天天气", "今天大风"), 0.5)
        self.assertEqual(cer.cer("", ""), 0.0)

    def test_refine_valid_output(self):
        # Legitimate short, English, and normal speech containing refusal-like words
        for text, cands in [
            ("好", ["好"]),
            ("嗯", ["嗯嗯"]),
            ("好", ["好", "好谢谢观看订阅点赞"]),
            ("OK", ["OK"]),
            ("886", ["886"]),
            ("抱歉，我来晚了", ["抱歉我来晚了"]),
            ("我无法理解他", ["我无法理解他"]),
            ("抱歉我不能来", ["抱歉我不能来"]),
            ("这个人不太靠谱", ["噎个人勿大靠谱"]),
        ]:
            self.assertTrue(refine._valid_output(text, cands), f"should accept {text!r}")

        # Near-empty outputs on long input and LLM meta-commentary / refusals
        for text, cands in [
            ("的", ["今末夜头阿拉去吃饭啷哉"]),
            ("。", ["好"]),
            ("这些识别结果都不太靠谱，请提供更准确的识别结果", ["乱码一堆啊"]),
            ("抱歉，我无法还原这句话的意思", ["乱码一堆啊"]),
            ("无法判断说话人的意思", ["乱码一堆啊"]),
            ("抱歉，听不清这段语音", ["乱码一堆啊"]),
        ]:
            self.assertFalse(refine._valid_output(text, cands), f"should reject {text!r}")

    def test_local_not_ready_error_message(self):
        fake_audio = SimpleNamespace(name="a.wav", seconds=1.0, pcm=b"", wav=b"")
        with patch.object(stt_core.PROVIDERS["whisper-ft"], "ready", lambda: False):
            r = stt_core.transcribe(fake_audio, "whisper-ft")
            self.assertNotIn("API key", r.error)
            self.assertIn("model", r.error)

    def test_whisper_split_pcm(self):
        short = np.ones(16000 * 25, dtype=np.float32)
        self.assertEqual(len(local_whisper_wu._split_pcm(short)), 1)

        for sec in (30.1, 35, 45, 59, 60, 95):
            n = int(sec * 16000)
            x = np.ones(n, dtype=np.float32) * 0.5
            x[n // 2 - 1600 : n // 2 + 1600] = 0.0
            chunks = local_whisper_wu._split_pcm(x)
            self.assertEqual(sum(len(c) for c in chunks), n)
            self.assertTrue(all(len(c) <= local_whisper_wu.MAX_SAMPLES for c in chunks))
            self.assertTrue(all(len(c) >= 8 * 16000 for c in chunks))


class TestLabeling(unittest.TestCase):
    def test_draft_skips_unready_and_jsonl_tolerates_partial(self):
        with tempfile.TemporaryDirectory() as d:
            msg_p, wav_p, drafts_p = (os.path.join(d, f) for f in ("messages.jsonl", "1_voice.wav", "drafts.json"))
            open(wav_p, "wb").close()
            with open(msg_p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"type": "voice", "voice_path": wav_p, "voice_ms": 1000, "time": 100}) + "\n")
                f.write('{"type": "voice", "half')
            with open(drafts_p, "w", encoding="utf-8") as f:
                json.dump({"1_voice.wav": {"wu-ft": "侬好", "wu-local": "侬好"}}, f)

            fake_providers = {  # whisper-ft missing: the clip must not be redone for it every 30 s
                "wu-ft": SimpleNamespace(ready=lambda: True, run=MagicMock(return_value="侬好")),
                "whisper-ft": SimpleNamespace(ready=lambda: False, run=MagicMock()),
            }
            audio_ctor = MagicMock()
            with patch.object(dataset, "MESSAGES", msg_p), \
                 patch.object(dataset, "DRAFTS", drafts_p), \
                 patch.object(dataset, "LABEL_DIR", d), \
                 patch.object(dataset, "extras", lambda: {}), \
                 patch("stt.core.PROVIDERS", fake_providers), \
                 patch("stt.audio.Audio", audio_ctor):
                self.assertEqual(dataset.draft(log=lambda *_: None), 0)
                self.assertEqual(audio_ctor.call_count, 0)
                # a retired drafter's entry stays in the file but is not shown
                self.assertIn("wu-local", dataset.drafts()["1_voice.wav"])
                self.assertEqual(dataset.shown_drafts(), {"1_voice.wav": {"wu-ft": "侬好"}})

    def test_chat_view_dynamic_name_voice_switch_and_db_close(self):
        with tempfile.TemporaryDirectory() as d:
            db_p = os.path.join(d, "bot.db")
            con = sqlite3.connect(db_p)
            con.executescript("""
                create table messages (id integer primary key, chat text, time real, role text, text text, image_path text, meta text);
                create table switches (chat text primary key, on_ integer, at real);
            """)
            con.execute("insert into messages values (1, 'g@chatroom', 105, 'assistant', '好的', null, null)")
            con.execute("insert into switches values ('g@chatroom', 1, 102)")
            con.commit()
            con.close()

            wav_p = os.path.join(d, "2_v.wav")
            open(wav_p, "wb").close()
            with open(os.path.join(d, "messages.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"chat": "g@chatroom", "chat_name": "测试群", "is_group": True,
                                    "sender": "u1", "sender_name": "甲", "is_self": False, "time": 100,
                                    "type": "text", "text": "hello", "local_id": 1, "reply": "never"}) + "\n")
                f.write(json.dumps({"chat": "g@chatroom", "chat_name": "测试群", "is_group": True,
                                    "sender": "u1", "sender_name": "甲", "is_self": False, "time": 102,
                                    "type": "voice", "voice_path": wav_p, "voice_ms": 1500, "local_id": 2,
                                    "reply": "never"}) + "\n")
            drafts_p = os.path.join(d, "drafts.json")
            with open(drafts_p, "w", encoding="utf-8") as f:
                json.dump({"2_v.wav": {"whisper-ft": "小助 开启"}}, f)

            closed = []
            real_connect = sqlite3.connect

            def tracked(*a, **kw):
                c = real_connect(*a, **kw)
                class Proxy:
                    def __getattr__(self, n): return getattr(c, n)
                    def close(self): closed.append(True); return c.close()
                return Proxy()

            cfg = {"group": {"names": ["小助"], "auto": True}, "reply": {"on_words": ["开启"], "off_words": ["关闭"]}}
            with patch.object(chat_view, "DATA", d), \
                 patch.object(chat_view, "DRAFTS", drafts_p), \
                 patch.object(chat_view, "LABELS", os.path.join(d, "labels.jsonl")), \
                 patch.object(chat_view, "VOICE_STT", os.path.join(d, "voice_stt.jsonl")), \
                 patch.object(chat_view, "load_config", lambda: cfg), \
                 patch("labeling.chat_view.sqlite3.connect", tracked):
                res = chat_view.chat("g@chatroom")
            self.assertEqual(len(closed), 1)
            self.assertEqual(res["bot_name"], "小助")
            self.assertTrue(res["group_auto"])
            bot_items = [i for i in res["items"] if i.get("bot")]
            self.assertEqual(bot_items[0]["sender"], "小助（机器人）")
            sw_items = [i for i in res["items"] if i["kind"] == "switch"]
            self.assertEqual(len(sw_items), 1)
            self.assertEqual(sw_items[0]["by"], "甲")
            self.assertTrue(sw_items[0]["on"])
            # Initial state came from messages.jsonl reply="never" (off) -> switched on at 102
            self.assertEqual([(p["on"], p["to"]) for p in res["periods"]], [(False, 102), (True, None)])

    def test_chat_view_voice_switch_uses_the_bots_transcript(self):
        """A spoken switch counts only if the transcript the bot used heard it: the STT result while on,
        the first [reply] command_stt draft while off. Other drafts must not draw phantom switches."""
        chat = "wxid_a"
        with tempfile.TemporaryDirectory() as d:
            wavs = []
            with open(os.path.join(d, "messages.jsonl"), "w", encoding="utf-8") as f:
                for i, (t, mode) in enumerate([(100, "always"), (110, "always"), (120, "always")]):
                    wavs.append(os.path.join(d, f"{i}_v.wav"))
                    open(wavs[-1], "wb").close()
                    f.write(json.dumps({"chat": chat, "chat_name": "友", "is_group": False, "sender": chat,
                                        "sender_name": "友", "is_self": False, "time": t, "type": "voice",
                                        "voice_path": wavs[-1], "voice_ms": 900, "local_id": i, "reply": mode}) + "\n")
            with open(os.path.join(d, "voice_stt.jsonl"), "w", encoding="utf-8") as f:
                for key, text in [("0_v.wav", "今天去哪"), ("1_v.wav", "关闭")]:
                    f.write(json.dumps({"chat": chat, "key": key, "text": text}) + "\n")
            drafts_p = os.path.join(d, "drafts.json")
            with open(drafts_p, "w", encoding="utf-8") as f:
                json.dump({"0_v.wav": {"wu-ft": "关闭"},                         # on: only the STT text counts
                           "2_v.wav": {"whisper-ft": "天气", "wu-ft": "开启"}}, f)  # off: whisper-ft first
            cfg = {"reply": {"on_words": ["开启"], "off_words": ["关闭"], "command_stt": ["whisper-ft", "wu-ft"]}}
            with patch.object(chat_view, "DATA", d), \
                 patch.object(chat_view, "DRAFTS", drafts_p), \
                 patch.object(chat_view, "LABELS", os.path.join(d, "labels.jsonl")), \
                 patch.object(chat_view, "VOICE_STT", os.path.join(d, "voice_stt.jsonl")), \
                 patch.object(chat_view, "load_config", lambda: cfg):
                res = chat_view.chat(chat)
        sw = [(i["time"], i["on"]) for i in res["items"] if i["kind"] == "switch"]
        self.assertEqual(sw, [(110, False)])
        self.assertEqual([(p["on"], p["to"]) for p in res["periods"]], [(True, 110), (False, None)])

    def test_chat_view_old_messages_default_state_and_bot_db_transcript(self):
        """Messages from before the "reply" field start in the config's default state, and a voice from
        before voice_stt.jsonl is read from what bot.db recorded."""
        chat = "wxid_b"
        with tempfile.TemporaryDirectory() as d:
            con = sqlite3.connect(os.path.join(d, "bot.db"))
            con.executescript("""
                create table messages (id integer primary key, chat text, time real, role text, text text, image_path text, meta text);
                create table switches (chat text primary key, on_ integer, at real);
            """)
            con.execute("insert into messages values (1, ?, 100, 'user', '[语音] 关闭', null, ?)",
                        (chat, json.dumps({"local_id": 7})))
            con.commit()
            con.close()
            wav_p = os.path.join(d, "7_v.wav")
            open(wav_p, "wb").close()
            with open(os.path.join(d, "messages.jsonl"), "w", encoding="utf-8") as f:  # no "reply" field
                f.write(json.dumps({"chat": chat, "chat_name": "老友", "is_group": False, "sender": chat,
                                    "sender_name": "老友", "is_self": False, "time": 100, "type": "voice",
                                    "voice_path": wav_p, "voice_ms": 900, "local_id": 7}) + "\n")
            cfg = {"chats": ["老友"], "reply": {"on_words": ["开启"], "off_words": ["关闭"]}}
            with patch.object(chat_view, "DATA", d), \
                 patch.object(chat_view, "DRAFTS", os.path.join(d, "drafts.json")), \
                 patch.object(chat_view, "LABELS", os.path.join(d, "labels.jsonl")), \
                 patch.object(chat_view, "VOICE_STT", os.path.join(d, "voice_stt.jsonl")), \
                 patch.object(chat_view, "load_config", lambda: cfg):
                res = chat_view.chat(chat)
        self.assertEqual([(i["time"], i["on"]) for i in res["items"] if i["kind"] == "switch"], [(100, False)])
        self.assertEqual([(p["on"], p["to"]) for p in res["periods"]], [(True, 100), (False, None)])

    def test_label_handler_400_audio_unquote_and_concurrent_glossary(self):
        def call(method, path, body=b""):
            h = object.__new__(label.Handler)
            h.path = path
            h.headers = {"Content-Length": str(len(body))}
            h.rfile, h.wfile = io.BytesIO(body), io.BytesIO()
            resp = {"code": None}
            h.send_response = lambda c: resp.__setitem__("code", c)
            h.send_header = lambda k, v: None
            h.end_headers = lambda: None
            getattr(h, f"do_{method}")()
            return resp["code"], h.wfile.getvalue()

        with tempfile.TemporaryDirectory() as d:
            gloss_p = os.path.join(d, "glossary.json")
            with patch.object(dataset, "LABEL_DIR", d), patch.object(dataset, "GLOSSARY", gloss_p):
                self.assertEqual(call("POST", "/api/label", b"{bad")[0], 400)
                self.assertEqual(call("POST", "/api/label", b"[]")[0], 400)
                def save(i):
                    b = json.dumps({"entries": [{"wu": f"词{i}", "mandarin": "意", "avoid": []}]}).encode()
                    return call("POST", "/api/glossary", b)[0]
                with ThreadPoolExecutor(6) as pool:
                    self.assertTrue(all(c == 200 for c in pool.map(save, range(16))))

                wav_p = os.path.join(d, "my clip.wav")
                with open(wav_p, "wb") as f:
                    f.write(b"RIFFWAVE")
                with patch.object(label, "voices", lambda: [{"key": "my clip.wav", "wav": wav_p}]):
                    code, data = call("GET", "/audio/my%20clip.wav?t=1")
                    self.assertEqual((code, data), (200, b"RIFFWAVE"))
