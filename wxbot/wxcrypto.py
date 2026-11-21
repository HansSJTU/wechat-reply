"""Decryption helpers for WeChat 4.x (macOS) local data.

- SQLCipher 4 databases (AES-256-CBC, HMAC-SHA512, 4096-byte pages, 80 reserved),
  including pages still sitting in the -wal file.
- V2 image .dat files (AES-128-ECB head + plain middle + XOR tail), and the HEVC "wxgf"
  pictures inside them.
"""
import os
import re
import shutil
import struct
import subprocess

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

PAGE = 4096
RESERVE = 80
SQLITE_HEADER = b"SQLite format 3\x00"
V2_MAGIC = b"\x07\x08V2\x08\x07"
IMAGE_MAGICS = {
    b"\xff\xd8\xff": "jpg",
    b"\x89PNG": "png",
    b"GIF8": "gif",
    b"wxgf": "wxgf",
    b"RIFF": "webp",
}


def page1_ok(enc_key: bytes, page1: bytes) -> bool:
    """True if enc_key authenticates page 1 (HMAC-SHA512 over ciphertext+IV+pgno)."""
    import hashlib, hmac
    mac_salt = bytes(b ^ 0x3A for b in page1[:16])
    mac_key = hashlib.pbkdf2_hmac("sha512", enc_key, mac_salt, 2, 32)
    h = hmac.new(mac_key, page1[16 : PAGE - RESERVE + 16], hashlib.sha512)
    h.update(struct.pack("<I", 1))
    return hmac.compare_digest(h.digest(), page1[PAGE - RESERVE + 16 : PAGE])


def key_for_db(db_path: str, keys: dict, account_dir: str) -> bytes:
    """Page key for a database: the verified one from wechat_keys.json, else derived
    from the account's raw key (covers databases created later, e.g. message_1.db)."""
    import hashlib
    rel = os.path.relpath(db_path, account_dir)
    if rel in keys.get("databases", {}):
        return bytes.fromhex(keys["databases"][rel]["enc_key"])
    with open(db_path, "rb") as f:
        page1 = f.read(PAGE)
    k = hashlib.pbkdf2_hmac("sha512", bytes.fromhex(keys["raw_key"]), page1[:16], keys["kdf"]["rounds"], 32)
    if not page1_ok(k, page1):
        raise ValueError(f"raw key does not open {rel}; re-capture keys")
    keys.setdefault("databases", {})[rel] = {"enc_key": k.hex(), "from": "raw_key"}
    return k


def decrypt_page(key: bytes, page: bytes, pgno: int) -> bytes:
    offset = 16 if pgno == 1 else 0
    iv = page[PAGE - RESERVE : PAGE - RESERVE + 16]
    dec = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    plain = dec.update(page[offset : PAGE - RESERVE]) + dec.finalize()
    head = SQLITE_HEADER if pgno == 1 else b""
    return head + plain + page[PAGE - RESERVE :]


def wal_pages(wal_path: str):
    """Latest committed copy of each page in the WAL, as {pgno: raw_page}."""
    if not os.path.exists(wal_path):
        return {}, None
    with open(wal_path, "rb") as f:
        wal = f.read()
    if len(wal) < 32:
        return {}, None
    salt = wal[16:24]
    pages, pending, db_size = {}, {}, None
    pos = 32
    while pos + 24 + PAGE <= len(wal):
        pgno, commit = struct.unpack(">II", wal[pos : pos + 8])
        if wal[pos + 8 : pos + 16] != salt:
            break
        pending[pgno] = wal[pos + 24 : pos + 24 + PAGE]
        if commit:
            pages.update(pending)
            pending = {}
            db_size = commit
        pos += 24 + PAGE
    return pages, db_size


def decrypt_db(db_path: str, key: bytes) -> bytes:
    """Return a plaintext SQLite image of db_path (+ its -wal)."""
    with open(db_path, "rb") as f:
        raw = f.read()
    pages = {i + 1: raw[i * PAGE : (i + 1) * PAGE] for i in range(len(raw) // PAGE)}
    wal, db_size = wal_pages(db_path + "-wal")
    pages.update(wal)
    n = db_size or max(pages)
    out = bytearray()
    for pgno in range(1, n + 1):
        page = pages.get(pgno)
        out += decrypt_page(key, page, pgno) if page else bytes(PAGE)
    # Mark the plaintext file as rollback-journal mode so sqlite3 opens it standalone.
    out[18:20] = b"\x01\x01"
    return bytes(out)


def parse_v2(data: bytes):
    if not data.startswith(V2_MAGIC):
        return None
    aes_size, xor_size = struct.unpack("<II", data[6:14])
    aes_len = aes_size - aes_size % 16 + 16
    body = data[15:]
    return body[:aes_len], body[aes_len : len(body) - xor_size], body[len(body) - xor_size :]


def image_kind(head: bytes):
    for magic, kind in IMAGE_MAGICS.items():
        if head.startswith(magic):
            return kind
    return None


def aes_ecb_decrypt(key: bytes, data: bytes) -> bytes:
    dec = Cipher(algorithms.AES(key), modes.ECB()).decryptor()
    return dec.update(data) + dec.finalize()


def wxgf_to_jpeg(data: bytes) -> bytes | None:
    """WeChat's own HEVC image container ("wxgf") -> JPEG via ffmpeg, or None.

    After the header come one or two raw HEVC streams, each starting at a VPS NAL unit; with
    two, the small one is the alpha mask and the larger one the picture.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not data.startswith(b"wxgf") or len(data) < 8:
        return None
    vps = [m.start() for m in re.finditer(rb"\x00\x00\x00\x01\x40\x01", data[data[4]:])]
    if not vps:
        return None
    bounds = list(zip(vps, vps[1:] + [len(data) - data[4]]))
    s, e = max(bounds, key=lambda b: b[1] - b[0])
    r = subprocess.run([ffmpeg, "-nostdin", "-loglevel", "error", "-f", "hevc", "-i", "-",
                        "-frames:v", "1", "-q:v", "2", "-f", "image2", "-c:v", "mjpeg", "-"],
                       input=data[data[4]:][s:e], capture_output=True, timeout=30)
    return r.stdout if r.returncode == 0 and r.stdout.startswith(b"\xff\xd8") else None


class MissingImageKey(ValueError):
    """A V2 image needs the AES key from secrets/wechat_keys.json; retrying cannot help."""


def decrypt_image(data: bytes, aes_key: bytes, xor_key: int):
    parts = parse_v2(data)
    if parts is None:  # legacy single-byte XOR format
        return bytes(b ^ xor_key for b in data)
    if not aes_key:
        raise MissingImageKey("missing image aes_key (run tools/derive_image_key.py)")
    aes_part, mid, xor_part = parts
    if not aes_part or len(aes_part) % 16 != 0:
        raise ValueError("invalid or truncated V2 image AES block")
    head = aes_ecb_decrypt(aes_key, aes_part)
    pad = head[-1]
    if not (1 <= pad <= 16 and head[-pad:] == bytes([pad]) * pad):
        raise ValueError(f"invalid PKCS7 padding ({pad}) while decrypting V2 image")
    head = head[:-pad]
    return head + mid + bytes(b ^ xor_key for b in xor_part)
