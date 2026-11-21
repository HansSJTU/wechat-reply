"""config.toml, with config.local.toml (gitignored: chats, location, prompts) merged on top key by key."""
import os
import tomllib

from .paths import ROOT


def _merge(base, over):
    for k, v in over.items():
        base[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def load_config(path=os.path.join(ROOT, "config.toml")):
    """`path`, then `config.local.toml` next to it on top."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    local = os.path.join(os.path.dirname(os.path.abspath(path)), "config.local.toml")
    if os.path.exists(local):
        with open(local, "rb") as f:
            cfg = _merge(cfg, tomllib.load(f))
    return cfg


def switch_words(cfg) -> tuple[list[str], list[str]]:
    """(on_words, off_words) of [reply]; unmute_words / mute_words are the old names."""
    r = cfg.get("reply", {})
    return (r.get("on_words", r.get("unmute_words", ["开启", "打开"])),
            r.get("off_words", r.get("mute_words", ["关闭"])))
