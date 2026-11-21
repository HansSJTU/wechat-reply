#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "websockets>=13", "sherpa-onnx"]
# ///
"""Compare speech-to-text providers on the same clips.

  uv run stt/cli.py                       every data/media/*voice.wav, every provider with a key
  uv run stt/cli.py a.m4a b.wav           specific files (any format ffmpeg reads)
  uv run stt/cli.py -p whisper-ft,wu-ft a.wav
  uv run stt/cli.py --list                providers and whether their key is set
  uv run stt/cli.py -e "期望1" -e "期望2" a.wav b.wav
                                          reference transcripts, paired with files in order (CER)

Ends with a per-provider summary (success, mean/max latency, mean CER).
Every result, with its latency, is appended to data/stt_results.jsonl.
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stt import PROVIDERS, available, transcribe  # noqa: E402
from stt.audio import Audio  # noqa: E402
from stt.cer import cer  # noqa: E402
from stt.keys import KEYS_FILE  # noqa: E402
from stt.paths import DATA  # noqa: E402

OUT = os.path.join(DATA, "stt_results.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("-p", "--providers", help="comma-separated; default: all with keys")
    ap.add_argument("-e", "--expect", action="append", default=[],
                    help="reference transcript; repeat once per file, in the same order")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, p in PROVIDERS.items():
            print(f"  {'✓' if p.ready() else '·'} {n:16} {p.label}")
        print(f"\nkeys: env vars or {os.path.relpath(KEYS_FILE)}")
        return

    names = args.providers.split(",") if args.providers else available()
    unknown = [n for n in names if n not in PROVIDERS]
    if unknown:
        sys.exit(f"unknown provider(s): {', '.join(unknown)}  (see --list)")
    if not names:
        sys.exit(f"no API keys configured — fill in {os.path.relpath(KEYS_FILE)} (see --list)")
    files = args.files or sorted(glob.glob(os.path.join(DATA, "media", "*voice.wav")))
    if not files:
        sys.exit("no input files")
    if args.expect and len(args.expect) != len(files):
        sys.exit(f"{len(args.expect)} -e references for {len(files)} files; give one per file")

    results = []
    with open(OUT, "a") as log, ThreadPoolExecutor(max_workers=len(names)) as pool:
        for i, path in enumerate(files):
            audio = Audio(path)
            expect = args.expect[i] if args.expect else None
            print(f"\n■ {audio.name}  ({audio.seconds:.1f}s)" + (f"  ref: {expect}" if expect else ""))
            for r in pool.map(lambda n: transcribe(audio, n), names):
                score = ""
                if expect is not None and not r.error:
                    r.extra["cer"] = round(cer(expect, r.text), 3)
                    score = f"  CER {r.extra['cer']:4.0%}"
                body = r.text if not r.error else f"✗ {r.error}"
                print(f"  {r.provider:16} {r.latency:5.2f}s{score}  {body}")
                if "llm_latency" in r.extra:
                    print(f"  {'':16} = asr {r.extra['asr_latency']:.2f}s + llm {r.extra['llm_latency']:.2f}s"
                          + (f"  (llm failed: {r.extra['refine_error']})" if "refine_error" in r.extra else ""))
                    for name, cand in r.extra.get("candidates", {}).items():
                        print(f"  {'':16}   · {name:12} {cand}")
                rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "path": os.path.abspath(path),
                       "expect": expect, **r.__dict__}
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                results.append(r)
    summarize(names, results)


def summarize(names, results):
    print(f"\n{'provider':16} {'ok':>5} {'mean s':>7} {'max s':>6} {'mean CER':>9}")
    for n in names:
        rs = [r for r in results if r.provider == n]
        ok = [r for r in rs if not r.error]
        lat = [r.latency for r in ok]
        cers = [r.extra["cer"] for r in ok if "cer" in r.extra]
        mean_lat = f"{sum(lat) / len(lat):7.2f}" if lat else f"{'-':>7}"
        max_lat = f"{max(lat):6.2f}" if lat else f"{'-':>6}"
        mean_cer = f"{sum(cers) / len(cers):9.0%}" if cers else f"{'-':>9}"
        print(f"{n:16} {len(ok):>2}/{len(rs):<2} {mean_lat} {max_lat} {mean_cer}")


if __name__ == "__main__":
    main()
