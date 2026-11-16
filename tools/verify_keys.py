#!/usr/bin/env python3
"""Check captured keys (hook_cc.py output) against every encrypted database.

Writes secrets/wechat_keys.json: {"databases": {relpath: {"enc_key": hex}}, "raw_key": hex,
"kdf": {...}}. raw_key is the account's SQLCipher passphrase (from the 256000-round PBKDF call
WeChat makes when it opens a database); it derives the key of every database, including ones
created later. Keys already in the file are kept. Stdlib only; never prints key material.
"""
import argparse
import glob
import hashlib
import hmac
import json
import os
import struct

PAGE, RESERVE = 4096, 80


def page1_ok(enc_key, page1):
    mac_salt = bytes(b ^ 0x3A for b in page1[:16])
    mac_key = hashlib.pbkdf2_hmac("sha512", enc_key, mac_salt, 2, 32)
    h = hmac.new(mac_key, page1[16 : PAGE - RESERVE + 16], hashlib.sha512)
    h.update(struct.pack("<I", 1))
    return hmac.compare_digest(h.digest(), page1[PAGE - RESERVE + 16 : PAGE])


ap = argparse.ArgumentParser()
ap.add_argument("--account-dir", required=True)
ap.add_argument("--captures", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

dbs = {}
for p in sorted(glob.glob(os.path.join(a.account_dir, "db_storage", "**", "*.db"), recursive=True)):
    with open(p, "rb") as f:
        page1 = f.read(PAGE)
    if len(page1) == PAGE and not page1.startswith(b"SQLite format 3\0"):
        dbs[os.path.relpath(p, a.account_dir)] = page1

aes, pbkdf = set(), []
with open(a.captures) as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if parts[0] == "aes":
            aes.add(bytes.fromhex(parts[1]))
        elif parts[0] == "pbkdf":
            pbkdf.append((bytes.fromhex(parts[1]), bytes.fromhex(parts[2]), int(parts[3])))
print(f"databases: {len(dbs)}; captured aes keys: {len(aes)}, pbkdf calls: {len(pbkdf)}")

found = {}
for rel, page1 in dbs.items():
    for k in aes:
        if page1_ok(k, page1):
            found[rel] = {"enc_key": k.hex()}
            break

# The account passphrase (raw key): a captured PBKDF password that opens a database. It also
# derives keys for databases not touched during capture.
raw = None
for pw, salt, rounds in pbkdf:
    if rounds < 1000:
        continue
    rel = next((r for r, page1 in dbs.items() if page1[:16] == salt), None)
    if rel and page1_ok(hashlib.pbkdf2_hmac("sha512", pw, salt, rounds, 32), dbs[rel]):
        raw = (pw, rounds)
        break
missing = [r for r in dbs if r not in found]
if raw:
    pw, rounds = raw
    for rel in list(missing):
        k = hashlib.pbkdf2_hmac("sha512", pw, dbs[rel][:16], rounds, 32)
        if page1_ok(k, dbs[rel]):
            found[rel] = {"enc_key": k.hex(), "from": "raw_key"}
            missing.remove(rel)

print(f"verified: {len(found)}/{len(dbs)}; raw key: {'found' if raw else 'NOT found (re-capture from WeChat launch)'}")
for rel in dbs:
    print(("  ok   " if rel in found else "  MISS ") + rel)

prev = {}
if os.path.exists(a.out):
    with open(a.out) as f:
        prev = json.load(f)
prev.setdefault("databases", {}).update(found)
if raw:
    prev["raw_key"] = raw[0].hex()
    prev["kdf"] = {"alg": "pbkdf2-sha512", "rounds": raw[1]}
with open(a.out, "w") as f:
    json.dump(prev, f, indent=2)
os.chmod(a.out, 0o600)
print(f"wrote {a.out}")
