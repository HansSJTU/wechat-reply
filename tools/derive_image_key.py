# /// script
# requires-python = ">=3.10"
# dependencies = ["cryptography"]
# ///
"""Derive the V2 image key offline and store it in secrets/wechat_keys.json.

WeChat 4.x (macOS): aes_key = md5(str(uin) + wxid).hexdigest()[:16], xor_key = uin & 0xff.
uin comes from file names under app_data/net/kvcomm/, wxid from the account folder
(without its 4-hex suffix). Every combination is checked against a real V2 .dat file.
"""
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "wxbot"))
from wxcrypto import V2_MAGIC, aes_ecb_decrypt, decrypt_image, image_kind, parse_v2  # noqa: E402

from private_io import write_private_json  # noqa: E402  (tools/, the script's own directory)

DOCS = os.path.expanduser("~/Library/Containers/com.tencent.xinWeChat/Data/Documents")
KEYS = os.path.join(HERE, "..", "secrets", "wechat_keys.json")

accs = glob.glob(DOCS + "/xwechat_files/wxid_*/")
if not accs:
    sys.exit(f"no WeChat account under {DOCS}")
acc = max(accs, key=lambda p: os.stat(p + "db_storage").st_mtime)
folder = os.path.basename(acc.rstrip("/"))
wxids = {folder, re.sub(r"_[0-9a-f]{4}$", "", folder)}
uins = set()
for p in glob.glob(DOCS + "/app_data/net/kvcomm/*"):
    uins.update(re.findall(r"_(\d{6,12})_", "_" + os.path.basename(p) + "_"))

sample = None
for p in glob.glob(acc + "msg/attach/**/*.dat", recursive=True):
    try:
        with open(p, "rb") as f:
            if f.read(len(V2_MAGIC)) == V2_MAGIC:
                sample = p
                break
    except OSError:
        continue
if not sample:
    sys.exit("need at least one received image (.dat V2) under msg/attach")
with open(sample, "rb") as f:
    data = f.read()
aes_part = parse_v2(data)[0]

for uin in sorted(uins):
    for w in wxids:
        key = hashlib.md5((uin + w).encode()).hexdigest()[:16].encode()
        if image_kind(aes_ecb_decrypt(key, aes_part[:16])):
            xor = int(uin) & 0xFF
            img = decrypt_image(data, key, xor)
            with open(KEYS) as f:
                keys = json.load(f)
            keys["image"] = {"aes_key": key.decode(), "xor_key": xor}
            write_private_json(KEYS, keys)
            print(f"image key verified ({image_kind(img)}), saved to {os.path.normpath(KEYS)}")
            sys.exit()
sys.exit("no uin/wxid combination decrypted the sample")
