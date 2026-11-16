"""In-memory plaintext mirror of a live, encrypted WeChat database.

refresh() re-reads the main file only when its mtime/size changed, re-reads the
(small) -wal every time, and decrypts only pages whose ciphertext differs from
what was applied last time. connect() hands out a private in-memory sqlite copy,
so plaintext never touches the disk.
"""
import os
import sqlite3

from .wxcrypto import PAGE, decrypt_page, wal_pages


class LiveDB:
    def __init__(self, path: str, key: bytes):
        self.path, self.key = path, key
        self._main_sig = None
        self._main = {}
        self._applied = {}
        self.plain = bytearray()

    def refresh(self) -> bool:
        st = os.stat(self.path)
        sig = (st.st_mtime_ns, st.st_size)
        if sig != self._main_sig:
            with open(self.path, "rb") as f:
                raw = f.read()
            self._main = {i + 1: raw[i * PAGE : (i + 1) * PAGE] for i in range(len(raw) // PAGE)}
            self._main_sig = sig
        wal, db_size = wal_pages(self.path + "-wal")
        n = db_size or (max(self._main) if self._main else 0)

        changed = False
        if len(self.plain) != n * PAGE:
            if len(self.plain) > n * PAGE:
                del self.plain[n * PAGE :]
                for p in [p for p in self._applied if p > n]:
                    del self._applied[p]
            else:
                self.plain.extend(bytes(n * PAGE - len(self.plain)))
            changed = True
        for pgno in range(1, n + 1):
            c = wal.get(pgno) or self._main.get(pgno)
            if c is None:
                continue
            prev = self._applied.get(pgno)
            if prev is c or prev == c:
                continue
            self.plain[(pgno - 1) * PAGE : pgno * PAGE] = decrypt_page(self.key, c, pgno)
            self._applied[pgno] = c
            changed = True
            if pgno == 1:
                self.plain[18:20] = b"\x01\x01"  # rollback-journal mode for the standalone copy
        return changed

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(":memory:")
        if self.plain:
            con.deserialize(bytes(self.plain))
        return con
