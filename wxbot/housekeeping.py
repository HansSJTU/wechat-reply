"""Keep data/ bounded: rotate the launchd logs and expire old media ([housekeeping] in config.toml).

launchd opens data/<name>.out / .log once and hands them to the process as stdout / stderr, so
nothing outside the process can rotate them. Here the process itself moves a log that grew past
`log_max_mb` to .1 (… .`log_keep`) and points its descriptor at a fresh file.

data/media holds the images the LLM is shown and every voice clip. Voice clips are labeling and
fine-tuning material (exports point at them), so they are kept unless `voice_days` is set, and a
labeled clip is never deleted.
"""
import fcntl
import json
import os
import stat
import sys
import threading
import time

from .log import log
from .paths import DATA

MEDIA = os.path.join(DATA, "media")
LABELS = os.path.join(DATA, "labels", "labels.jsonl")
DAY = 86400.0


def _fd_path(fd):
    try:
        return os.fsdecode(fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024)).split(b"\0", 1)[0])
    except (AttributeError, OSError):  # F_GETPATH is macOS-only
        return None


def _shift(path, keep):
    for i in range(keep - 1, 0, -1):
        if os.path.exists(f"{path}.{i}"):
            os.replace(f"{path}.{i}", f"{path}.{i + 1}")
    os.replace(path, f"{path}.1")


def rotate_std_logs(max_bytes, keep=3, fds=(1, 2)):
    """Rotate each regular file behind `fds` that is at least `max_bytes`. Returns the rotated paths."""
    keep = max(1, int(keep))
    groups = {}  # stdout and stderr may share one file
    for fd in fds:
        try:
            st = os.fstat(fd)
        except OSError:
            continue
        if stat.S_ISREG(st.st_mode) and st.st_size >= max_bytes:
            groups.setdefault((st.st_dev, st.st_ino), []).append(fd)
    rotated = []
    for (dev, ino), group in groups.items():
        path = _fd_path(group[0])
        try:
            if not path or (os.stat(path).st_dev, os.stat(path).st_ino) != (dev, ino):
                continue  # renamed or replaced meanwhile: not ours to move
        except FileNotFoundError:
            continue
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass
        _shift(path, keep)
        new = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            for fd in group:
                os.dup2(new, fd)
        finally:
            os.close(new)
        rotated.append(path)
    return rotated


def labeled_keys(path=LABELS):
    """Voice file names that have a label (any status): never deleted."""
    keys = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    keys.add(json.loads(line)["key"])
                except (ValueError, KeyError, TypeError):
                    continue
    except FileNotFoundError:
        pass
    return keys


def clean_media(media_dir=MEDIA, image_days=120, voice_days=0, protected=(), now=None):
    """Delete images older than `image_days` and unlabeled voice clips older than `voice_days` (0 or less:
    keep), plus .silk / .pcm scraps a crash left mid-conversion. Returns the number of files removed."""
    now = time.time() if now is None else now
    removed = 0
    try:
        entries = list(os.scandir(media_dir))
    except FileNotFoundError:
        return 0
    for e in entries:
        name = e.name
        if name.endswith((".silk", ".pcm")):
            days = 1
        elif name.endswith("_voice.wav"):
            days = 0 if name in protected else voice_days
        else:
            days = image_days
        try:
            if not days or days <= 0 or not e.is_file() or now - e.stat().st_mtime < days * DAY:
                continue
            os.unlink(e.path)
            removed += 1
        except FileNotFoundError:
            continue
    return removed


def start(cfg, media=True, interval=600.0):
    """Background thread: check the logs every `interval` seconds; expire media at start and then daily."""
    h = cfg.get("housekeeping", {})
    max_bytes = int(float(h.get("log_max_mb", 5)) * (1 << 20))
    keep = int(h.get("log_keep", 3))

    def loop():
        last_media = 0.0
        while True:
            try:
                for p in rotate_std_logs(max_bytes, keep):
                    log(f"housekeeping: rotated {os.path.basename(p)}")
            except Exception as e:
                log(f"housekeeping: log rotation failed: {type(e).__name__}: {e}")
            if media and time.time() - last_media >= DAY:
                last_media = time.time()
                try:
                    n = clean_media(image_days=float(h.get("image_days", 120)),
                                    voice_days=float(h.get("voice_days", 0)), protected=labeled_keys())
                    if n:
                        log(f"housekeeping: removed {n} old file(s) from data/media")
                except Exception as e:
                    log(f"housekeeping: media cleanup failed: {type(e).__name__}: {e}")
            time.sleep(interval)

    threading.Thread(target=loop, name="housekeeping", daemon=True).start()
