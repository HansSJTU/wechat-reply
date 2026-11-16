"""Local Wu-dialect ASR: WenetSpeech-Wu U2++ Conformer CTC (ASLP-lab), sherpa-onnx int8 export.

~130 MB, CPU only, no API key. Trained on ~8000 h of Wu (mostly 上海话/苏州话 etc.); outputs Wu
characters (侬, 搿, 阿拉), not Mandarin. Model files: models/wu-u2pp-int8/ from
https://huggingface.co/csukuangfj2/sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03
"""
import array
import os

from .audio import RATE, Audio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "wu-u2pp-int8")
_rec = None


def ready() -> bool:
    if not os.path.exists(os.path.join(MODEL_DIR, "model.int8.onnx")):
        return False
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return True


def _recognizer():
    global _rec
    if _rec is None:
        import sherpa_onnx
        _rec = sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
            model=os.path.join(MODEL_DIR, "model.int8.onnx"),
            tokens=os.path.join(MODEL_DIR, "tokens.txt"),
            num_threads=4,
        )
    return _rec


def transcribe(audio: Audio) -> str:
    samples = array.array("h", audio.pcm)  # 16-bit little-endian mono
    rec = _recognizer()
    s = rec.create_stream()
    s.accept_waveform(RATE, [x / 32768.0 for x in samples])
    rec.decode_stream(s)
    return s.result.text.strip()
