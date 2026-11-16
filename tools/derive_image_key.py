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
from wxcrypto import aes_ecb_decrypt, decrypt_image, image_kind, parse_v2  # noqa: E402

DOCS = os.path.expanduser("~/Library/Containers/com.tencent.xinWeChat/Data/Documents")
KEYS = os.path.join(HERE, "..", "secrets", "wechat_keys.json")

acc = glob.glob(DOCS + "/xwechat_files/wxid_*/")[0]
folder = os.path.basename(acc.rstrip("/"))
wxids = {folder, re.sub(r"_[0-9a-f]{4}$", "", folder)}
uins = set()
for p in glob.glob(DOCS + "/app_data/net/kvcomm/*"):
    uins.update(re.findall(r"_(\d{6,12})_", "_" + os.path.basename(p) + "_"))

samples = [p for p in glob.glob(acc + "msg/attach/**/*.dat", recursive=True)]
sample = next((p for p in samples if parse_v2(open(p, "rb").read())), None)
if not sample:
    sys.exit("need at least one received image (.dat V2) under msg/attach")
data = open(sample, "rb").read()
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
            with open(KEYS, "w") as f:
                json.dump(keys, f, indent=2)
            os.chmod(KEYS, 0o600)
            print(f"image key verified ({image_kind(img)}), saved to {os.path.normpath(KEYS)}")
            sys.exit()
sys.exit("no uin/wxid combination decrypted the sample")
