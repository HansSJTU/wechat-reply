"""Normalize any input audio to 16 kHz mono 16-bit PCM, which every provider accepts."""
import os
import shutil
import subprocess
import tempfile
import wave

RATE = 16000


class Audio:
    """One clip, decoded once and shared by every provider."""

    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        self.wav = _to_wav(path)  # bytes of a complete .wav file
        with wave.open(_BytesFile(self.wav)) as w:
            self.pcm = w.readframes(w.getnframes())
            self.seconds = w.getnframes() / w.getframerate()


def _to_wav(path: str) -> bytes:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "a.wav")
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path,
                   "-ac", "1", "-ar", str(RATE), "-sample_fmt", "s16", out]
        else:  # macOS built-in
            cmd = ["afconvert", "-f", "WAVE", "-d", f"LEI16@{RATE}", "-c", "1", path, out]
        subprocess.run(cmd, check=True, capture_output=True)
        with open(out, "rb") as f:
            return f.read()


class _BytesFile:
    """Minimal seekable file over bytes (wave.open needs read/seek/tell)."""

    def __init__(self, b: bytes):
        self.b, self.i = b, 0

    def read(self, n=-1):
        end = len(self.b) if n < 0 else self.i + n
        out = self.b[self.i:end]
        self.i += len(out)
        return out

    def seek(self, i, whence=0):
        self.i = i if whence == 0 else (self.i + i if whence == 1 else len(self.b) + i)
        return self.i

    def tell(self):
        return self.i

    def close(self):
        pass
