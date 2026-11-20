#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["cryptography", "zstandard", "pilk", "httpx", "websockets>=13", "frida==17.26.3",
#                 "sherpa-onnx", "torch", "openai-whisper"]  # local wu-ft / whisper-ft for the STT combo
# ///
"""WeChat reply bot entry point (the LaunchAgent launcher runs this file; keep its name).

  uv run monitor.py [run] [--dry-run]   run the bot (reads config.toml)
  uv run monitor.py chats [query]       list chats to pick from
  uv run monitor.py send --to U --text T
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wxbot.main import main  # noqa: E402

if __name__ == "__main__":
    main()
