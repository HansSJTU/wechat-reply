#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyobjc-framework-Vision", "pyobjc-framework-Quartz", "numpy", "pillow"]
# ///
"""Timed on-screen text from a video with macOS Vision OCR (local, no network).

  uv run stt/bili_ocr.py VIDEO --crop x,y,w,h [--fps 5] > segments.jsonl

Samples the crop at --fps, OCRs only when the bright text pixels change, and prints one JSON line per
stretch of unchanged text: {"start", "end", "lines": [...]} (seconds; lines top to bottom).
"""
import argparse
import json
import subprocess
import sys

import numpy as np


def frames(video: str, crop: tuple[int, int, int, int], fps: float):
    x, y, w, h = crop
    cmd = ["ffmpeg", "-v", "error", "-i", video, "-vf", f"fps={fps},crop={w}:{h}:{x}:{y},format=gray",
           "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    i, size = 0, w * h
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield i / fps, np.frombuffer(buf, np.uint8).reshape(h, w)
        i += 1


def ocr(img: np.ndarray) -> list[str]:
    import Quartz
    import Vision
    from Foundation import NSData
    from PIL import Image
    import io
    b = io.BytesIO()
    Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2)).save(b, "PNG")
    data = NSData.dataWithBytes_length_(b.getvalue(), len(b.getvalue()))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLanguages_(["zh-Hans", "en-US"])
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    handler.performRequests_error_([req], None)
    obs = sorted(req.results() or [], key=lambda o: -o.boundingBox().origin.y)
    return [o.topCandidates_(1)[0].string() for o in obs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--crop", required=True, help="x,y,w,h in source pixels")
    ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--bright", type=int, default=200, help="text pixel threshold (white subtitles)")
    ap.add_argument("--dark", action="store_true", help="dark text on a light background")
    ap.add_argument("--change", type=float, default=0.004, help="fraction of pixels that must change")
    a = ap.parse_args()
    crop = tuple(int(v) for v in a.crop.split(","))
    cur, start, last_mask, t = None, 0.0, None, 0.0
    for t, img in frames(a.video, crop, a.fps):
        mask = img < 255 - a.bright if a.dark else img > a.bright
        if last_mask is not None and np.mean(mask != last_mask) < a.change:
            continue
        last_mask = mask
        lines = ocr(img) if mask.mean() > 0.0005 else []
        if lines != cur:
            if cur:
                print(json.dumps({"start": round(start, 2), "end": round(t, 2), "lines": cur}, ensure_ascii=False),
                      flush=True)
            cur, start = lines, t
    if cur:
        print(json.dumps({"start": round(start, 2), "end": round(t, 2), "lines": cur}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
