"""Speech-to-text over several cloud APIs behind one call.

    from stt import transcribe
    r = transcribe("data/media/2_voice.wav")
    print(r.text, r.error)

See stt/README.md for providers and keys.
"""
from .core import PROVIDERS, Result, available, transcribe

__all__ = ["PROVIDERS", "Result", "available", "transcribe"]
