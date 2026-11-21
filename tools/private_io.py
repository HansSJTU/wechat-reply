"""Files holding key material: owner-only (0600) from the moment they exist.

Creating with default permissions and chmod-ing afterwards leaves a window in which another local
process can open the file and keep reading through that descriptor. Standard library only, so lldb's
Python (hook_cc.py) can import it too.
"""
import json
import os


def create_private(path: str, flags: int = os.O_WRONLY | os.O_APPEND) -> int:
    """Open `path` (created if missing) as 0600 and return the descriptor."""
    fd = os.open(path, flags | os.O_CREAT, 0o600)
    os.fchmod(fd, 0o600)  # the mode argument only applies when the file is new
    return fd


def write_private_json(path: str, obj) -> None:
    """Atomically replace `path` with `obj` as JSON, never readable by others."""
    tmp = path + ".tmp"
    try:
        with os.fdopen(create_private(tmp, os.O_WRONLY | os.O_TRUNC), "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
