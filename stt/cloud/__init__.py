"""Cloud speech APIs, kept for comparison (`stt/cli.py`); the bot's `combo` uses only the local models.

Each module has `ready()` (keys configured) and one or more transcribe functions taking an `Audio`.
They are imported lazily by `stt/core.py`, so httpx / websockets are only needed when one is used.
Results of the comparisons: docs/stt-experiments.md.
"""
