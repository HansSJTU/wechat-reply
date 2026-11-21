"""SQLite state (data/bot.db): read cursors and the conversation history the LLM sees.

Native sends never appear in WeChat's own database, so the bot's replies are recorded here
and nowhere else.
"""
import json
import sqlite3
import threading
import time

SCHEMA = """
create table if not exists cursors (key text primary key, local_id integer not null);
create table if not exists messages (
  id integer primary key,
  chat text not null,
  time real not null,
  role text not null check (role in ('user', 'assistant')),
  text text not null default '',
  image_path text,
  meta text
);
create index if not exists messages_chat_time on messages (chat, time);
create table if not exists mutes (chat text primary key, until real not null);
create table if not exists history_floor (chat text primary key, since real not null);
create table if not exists switches (chat text primary key, on_ integer not null, at real not null);
"""


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("pragma journal_mode=wal")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def cursor(self, key):
        with self.lock:
            row = self.db.execute("select local_id from cursors where key = ?", (key,)).fetchone()
        return row and row[0]

    def set_cursor(self, key, local_id):
        with self.lock:
            self.db.execute("insert into cursors values (?, ?) on conflict(key) do update set "
                            "local_id = max(local_id, excluded.local_id)", (key, local_id))

    def add(self, chat, role, text, image_path=None, at=None, **meta):
        with self.lock:
            cur = self.db.execute("insert into messages (chat, time, role, text, image_path, meta) values (?,?,?,?,?,?)",
                                  (chat, at or time.time(), role, text, image_path,
                                   json.dumps(meta, ensure_ascii=False) if meta else None))
            return cur.lastrowid

    def history(self, chat, since):
        with self.lock:
            rows = self.db.execute("select id, time, role, text, image_path from messages "
                                   "where chat = ? and time >= ? order by id", (chat, since)).fetchall()
        return [dict(id=rid, time=t, role=r, text=x, image_path=i) for rid, t, r, x, i in rows]

    def muted_until(self, chat):
        with self.lock:
            row = self.db.execute("select until from mutes where chat = ?", (chat,)).fetchone()
        return row[0] if row else 0.0

    def set_mute(self, chat, until):
        """until = 0 unmutes."""
        with self.lock:
            self.db.execute("insert into mutes values (?, ?) on conflict(chat) do update set until = excluded.until",
                            (chat, until))

    def enabled(self, chat):
        """True / False once the chat was switched with 开启 / 关闭; None before (config decides)."""
        with self.lock:
            row = self.db.execute("select on_ from switches where chat = ?", (chat,)).fetchone()
        return None if row is None else bool(row[0])

    def is_on(self, chat, mode):
        """开启 / 关闭 if the chat was switched, else its configured reply mode decides."""
        s = self.enabled(chat)
        return mode in ("always", "mention") if s is None else s

    def set_enabled(self, chat, on, at=None):
        with self.lock:
            self.db.execute("insert into switches values (?, ?, ?) on conflict(chat) do update set "
                            "on_ = excluded.on_, at = excluded.at", (chat, int(on), at or time.time()))

    def history_floor(self, chat):
        """The LLM sees nothing older than this (set by 开启)."""
        with self.lock:
            row = self.db.execute("select since from history_floor where chat = ?", (chat,)).fetchone()
        return row[0] if row else 0.0

    def set_history_floor(self, chat, since):
        with self.lock:
            self.db.execute("insert into history_floor values (?, ?) on conflict(chat) do update set "
                            "since = excluded.since", (chat, since))

    def replies_since(self, chat, since):
        with self.lock:
            return self.db.execute("select count(*) from messages where chat = ? and role = 'assistant' "
                                   "and time >= ?", (chat, since)).fetchone()[0]

    def voices(self, chat, since):
        """Transcribed voices ("[语音] …") received after `since`: (count, time of the newest or 0)."""
        with self.lock:
            n, newest = self.db.execute("select count(*), max(time) from messages where chat = ? and role = 'user' "
                                        "and time > ? and json_extract(meta, '$.type') = 'voice' "
                                        "and instr(text, '[语音] ') > 0", (chat, since)).fetchone()
        return n, newest or 0.0

    def last_summary(self, chat):
        """Where the bot's last chat summary ended (meta.upto, just after the newest line it covered), 0 if none."""
        with self.lock:
            row = self.db.execute("select max(coalesce(json_extract(meta, '$.upto'), time)) from messages "
                                  "where chat = ? and role = 'assistant' and json_extract(meta, '$.summary') "
                                  "is not null", (chat,)).fetchone()
        return row[0] or 0.0

    def auto_replies(self, chat, since):
        """Unprompted group replies (meta.auto = the judge's kind) since `since`: [(time, kind)], oldest first."""
        with self.lock:
            rows = self.db.execute("select time, json_extract(meta, '$.auto') from messages where chat = ? "
                                   "and role = 'assistant' and time >= ? and json_extract(meta, '$.auto') "
                                   "is not null order by time", (chat, since)).fetchall()
        return [(t, k) for t, k in rows]

    def checkpoint(self):
        """Fold the WAL back into the main DB file (at shutdown). The connection stays open: worker
        threads may still record a reply that was just sent, and those writes must not fail."""
        with self.lock:
            try:
                self.db.execute("pragma wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
