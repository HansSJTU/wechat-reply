"""lldb script: capture CommonCrypto keys from a running WeChat for a while.

SQLCipher's CommonCrypto provider calls
  CCCryptorCreate(op, alg, opts, key=x3, keyLen=x4, iv, ref)      per page
  CCHmacInit(ctx, alg=x1, key=x2, keyLen=x3)                       per page
  CCKeyDerivationPBKDF(alg, pw=x1, pwLen=x2, salt=x3, saltLen=x4,
                       prf, rounds=x6, out=x7, outLen)             on open
Breakpoint callbacks log each distinct key and return False (auto-continue).

usage (arm64):
  sudo lldb -p <pid> -o "command script import hook_cc.py" \
       -o "hookrun <seconds> <out.tsv>" -o quit
"""
import os
import threading
import time

import lldb

from private_io import create_private  # same directory; `command script import` puts it on sys.path

OUT = None
SEEN = set()
_LOCK = threading.Lock()


def _read(frame, addr, n):
    err = lldb.SBError()
    data = frame.GetThread().GetProcess().ReadMemory(addr, n, err)
    return data if err.Success() else None


def _reg(frame, name):
    return frame.FindRegister(name).GetValueAsUnsigned()


def _log(line):
    with _LOCK:
        if line in SEEN:
            return
        SEEN.add(line)
        with open(OUT, "a") as f:
            f.write(line + "\n")


def on_create(frame, bp_loc, internal_dict):
    n = _reg(frame, "x4")
    if n in (16, 32):  # 32: SQLCipher page key; 16: image .dat AES key
        k = _read(frame, _reg(frame, "x3"), n)
        if k:
            _log(("aes\t" if n == 32 else "aes16\t") + k.hex())
    return False


def on_hmac(frame, bp_loc, internal_dict):
    n = _reg(frame, "x3")
    if n in (32, 64):
        k = _read(frame, _reg(frame, "x2"), n)
        if k:
            _log(f"hmac{_reg(frame, 'x1')}\t" + k.hex())
    return False


def on_pbkdf(frame, bp_loc, internal_dict):
    pw = _read(frame, _reg(frame, "x1"), min(_reg(frame, "x2"), 512))
    salt = _read(frame, _reg(frame, "x3"), min(_reg(frame, "x4"), 64))
    rounds = _reg(frame, "x6") & 0xFFFFFFFF
    if pw is not None and salt is not None:
        _log(f"pbkdf\t{pw.hex()}\t{salt.hex()}\t{rounds}")
    return False


def _open_out(path):
    global OUT
    OUT = path
    fd = create_private(OUT)  # 0600 from creation: key material is appended to it
    try:
        uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
        if uid and gid:
            os.fchown(fd, int(uid), int(gid))
    finally:
        os.close(fd)


def hookrun(debugger, command, result, internal_dict):
    """hookrun <seconds> <out>: capture from the already-attached process."""
    args = command.split()
    _open_out(args[1])
    target = debugger.GetSelectedTarget()
    _capture(debugger, target, target.GetProcess(), int(args[0]))


def hookwait(debugger, command, result, internal_dict):
    """hookwait <seconds> <out>: wait for WeChat to launch, attach, capture (sees DB opens)."""
    args = command.split()
    _open_out(args[1])
    target = debugger.CreateTarget("")
    err = lldb.SBError()
    print("waiting for WeChat to launch...", flush=True)
    process = target.Attach(lldb.SBAttachInfo("WeChat", True), err)
    if not err.Success():
        print("attach failed:", err.GetCString(), flush=True)
        return
    print(f"attached to pid {process.GetProcessID()}", flush=True)
    _capture(debugger, target, process, int(args[0]))


def _capture(debugger, target, process, secs):
    for name, fn in (("CCCryptorCreate", "on_create"), ("CCHmacInit", "on_hmac"),
                     ("CCKeyDerivationPBKDF", "on_pbkdf")):
        bp = target.BreakpointCreateByName(name)
        bp.SetScriptCallbackFunction("hook_cc." + fn)
        print(f"breakpoint {name}: {bp.GetNumLocations()} location(s)", flush=True)

    debugger.SetAsync(True)
    process.Continue()
    print(f"capturing for {secs}s -- open chats / scroll history / send a message now", flush=True)
    # Stop early once the account's raw key (a 256000-round PBKDF call) is in hand and
    # captures have gone quiet: long breakpoint sessions can wedge WeChat.
    t0 = last_change = time.time()
    last_n = 0
    while time.time() - t0 < secs:
        time.sleep(5)
        with _LOCK:
            n_seen = len(SEEN)
            have_raw = any(s.startswith("pbkdf\t") and int(s.rsplit("\t", 1)[1]) > 1000 for s in SEEN)
        if n_seen != last_n:
            last_n, last_change = n_seen, time.time()
        print(f"  {int(time.time() - t0)}s  distinct captures: {n_seen}", flush=True)
        if have_raw and time.time() - last_change > 10:
            print("raw key captured and quiet for 10s; stopping early", flush=True)
            break
    process.Stop()
    time.sleep(1)
    target.DeleteAllBreakpoints()
    process.Detach()
    with _LOCK:
        total = len(SEEN)
    print(f"detached; wrote {total} captures to {OUT}", flush=True)


def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand("command script add -f hook_cc.hookrun hookrun")
    debugger.HandleCommand("command script add -f hook_cc.hookwait hookwait")
