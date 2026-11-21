"""Speech-to-text behind one call.

    from stt import transcribe
    r = transcribe("data/media/2_voice.wav")
    print(r.text, r.error)

See stt/README.md for providers and keys. The registry (and with it httpx for the LLM step) is loaded on
first use, so `stt.paths`, `stt.audio` or `stt.local_wu` can be imported without the API clients.
"""
__all__ = ["PROVIDERS", "Result", "available", "transcribe"]


def __getattr__(name):
    if name in __all__:
        from . import core
        return getattr(core, name)
    raise AttributeError(f"module 'stt' has no attribute {name!r}")
