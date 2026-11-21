"""16 kHz mono int16 audio as numpy arrays, for cutting clips out of long recordings."""
import os
import subprocess
import wave

import numpy as np

RATE = 16000
HOP = RATE // 100  # 10 ms frames for loudness


def to_wav(src: str, dst: str) -> str:
    """Convert any audio/video file to 16 kHz mono wav at `dst` with ffmpeg, once (cached)."""
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", str(RATE), dst + ".tmp.wav"],
                       check=True)
        os.replace(dst + ".tmp.wav", dst)
    return dst


def read_pcm(path: str) -> np.ndarray:
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def write_wav(path: str, x: np.ndarray):
    with wave.open(path + ".tmp", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(x.astype(np.int16).tobytes())
    os.replace(path + ".tmp", path)


def frame_rms(x: np.ndarray) -> np.ndarray:
    """RMS of each 10 ms frame."""
    n = len(x) // HOP
    return np.sqrt((x[:n * HOP].astype(np.float64).reshape(n, HOP) ** 2).mean(1))


def speech_span(x: np.ndarray, pad: float = 0.25):
    """First/last stretch louder than -30 dB of the peak and 3× the noise floor (as on the labeling page)."""
    if len(x) // HOP < 5:
        return None
    rms = frame_rms(x)
    thr = max(rms.max() * 0.0316, 3 * np.percentile(rms, 10))
    run = np.convolve(rms > thr, np.ones(3), "same") >= 3
    idx = np.flatnonzero(run)
    if not len(idx):
        return None
    a = max(0, idx[0] * HOP - int(pad * RATE))
    b = min(len(x), (idx[-1] + 1) * HOP + int(pad * RATE))
    return a, b
