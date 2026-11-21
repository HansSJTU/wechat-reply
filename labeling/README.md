# labeling — voice labels for fine-tuning the local Wu models

Goal: ~30–60 min of WeChat voice labeled with **what was actually said, in Wu characters** (侬、阿拉、搿…,
not a Mandarin translation) to fine-tune the U2++ Wu model behind `wu-local` (CTC can only learn
sound→character, not translation; the LLM step still translates).

```
uv run labeling/label.py serve     # http://127.0.0.1:8765 — local only; audio never leaves the machine
scripts/install.sh label           # or keep it running in the background under launchd (see the main README)
uv run labeling/label.py draft     # pre-fill drafts (the local models) for every saved voice
uv run labeling/label.py export    # data/labels/wenet/{train,test}.jsonl, {"key","wav","txt"} for WeNet
```

| file | what |
|---|---|
| `label.py` | the page's HTTP server and the CLI above; `convert` (我听到的 → 余姚话) |
| `dataset.py` | clips, drafts, labels, 说法表, test split, WeNet export (stdlib only) |
| `llm.py` | one structured LLM call (GPT, Gemini, or the `agy` CLI) |
| `chat_view.py` | data for the chat viewer at `/chats` |
| `label_ui.html`, `chats_ui.html` | the two pages |
| `sources/` | one-off builders of extra training clips (videos, web audio, reading sheets); they write `data/labels/extra_clips*.jsonl`, which the page lists next to WeChat voices |

- Voices: every `type: voice` entry in `data/messages.jsonl` (what monitor.py saves). `serve` drafts new
  ones in the background every 30 s.
- Page: newest unlabeled first, audio autoplays, two drafts to start from, a field for the Wu transcript
  and an optional Mandarin meaning (pre-filled from earlier `-e` references). ⌘/Ctrl+Enter save & go to the clip below (Ctrl+Enter in the 我听到的 box converts instead),
  Esc replay, Ctrl+1/2 use a draft. Warns about characters missing from the model's 5946-char vocabulary
  (they can't be learned without growing the vocabulary).
- Files: `data/labels/drafts.json`, `data/labels/labels.jsonl` (append-only; last line per clip wins;
  status done/skip/bad/exclude). Export strips punctuation and lists out-of-vocabulary characters.
- **首尾空白** (waveform under the player): “去掉首尾空白” finds the first and last stretch of speech
  (louder than -30 dB of the peak and 3× the noise floor) and keeps it plus 0.25 s on each side; or drag
  on the waveform to choose the part to keep; click to play from there. The range is saved with the
  label (`"trim": [start, end]` seconds); replay plays only that part, and export writes the cut clip
  to `wenet/wav/` instead of pointing at the original. The original audio is never changed.
- **不使用 (exclude)**: keeps the transcript but puts the clip in neither train nor test (private content,
  test messages to the bot, …). Only `done` clips are exported; change a clip's status any time.
- **我听到的 → 转换** (above the transcript): type what you heard, mixing Wu characters, Mandarin and
  pinyin (“我们这个鱼要吃吗 要吃的我拿一点起lang”), press ⌃⏎ (plain ⏎ does nothing), and the chosen model (with the whole
  说法表 in the prompt) returns the 余姚话 in 说法表 spellings plus the Mandarin meaning, each with a
  “用这个” button. Pick the model next to the input (remembered in the browser; list in `CONVERT_MODELS`):
  `gpt-6.2-sol` (default, ~3–5 s) or `gemini-4-hydrogen` (~10–15 s, only if picked). Only Gemini can take
  the audio: ticking 带录音 also sends the clip (~1 min) and may fill in parts you missed — or invent some,
  so check by ear. Audio alone without your sentence is not usable: on 18 labeled clips Gemini scored CER
  69% (hydrogen) / 84% (flash) vs 56% for the whisper-wu draft, often with fluent made-up sentences.
  Same from the shell: `uv run labeling/label.py convert "…" [--model gpt-6.2-sol] [--key …]`.
- **说法表** (header button; `data/labels/glossary.json`): one chosen spelling per 余姚话 word, its
  Mandarin meaning, and spellings to flag. When a flagged spelling (e.g. the drafts' 上海话 伊/今朝, or a
  homophone like 几么) is in the transcript, the page offers a one-click replace. Select text in the
  transcript → “把选中的字加入说法表” to add an entry; every entry is also a button under the transcript
  that inserts it at the cursor. ~200 entries in the spellings of a 余姚城区话 study (今末=今天, 上末=昨天,
  渠=他/她, 首西=什么, 杂话=怎么样, 葛=的, 噎=这, 啷哉=了). Meant to double as a dictionary for the LLM step (a test with a hand-written Wu transcript of
  clip 18: deepseek-flash CER 73% → 54% with such a table in the prompt).
- Split: a fixed ~20% of clips by md5 of the file name is **test** — never train on it, it is how a
  fine-tuned model is judged against the original.

## Chat viewer (`/chats` on the labeling server)

http://127.0.0.1:8765/chats — a WeChat-style view of each watched chat (`labeling/chat_view.py` +
`labeling/chats_ui.html`, local only): the chat list with each chat's 开启/关闭 state, the messages and the
bot's replies, a strip of 开启/关闭 periods with a tick per switch, a marker in the conversation at every
switch, and under each voice the bot's final Mandarin, every recognizer's candidate (expand), the local
model drafts and the human label if any. Refreshes every 10 s.

## Reading sheets (`labeling/sources/read_sheets.py`)

Instead of transcribing what people said, write the text first and have people read it: sentences in
说法表 spellings (`data/read/sentences.tsv`: id, 余姚话, 普通话) go out as numbered sheets of 50, with a
pinyin-style reading hint per sentence (`data/read/readings.tsv`, words before characters) so readers
recognise unusual characters; they send one WeChat voice per sentence. `check` flags characters missing
from the model vocabulary, 说法表 avoid-spellings, odd lengths, characters without a 余姚 reading, and
lists unused 说法表 words; `sheets` writes `data/read/sheets/sheet-NN.txt` (+ `.json` with the ids).
Read speech is train-only; the test split stays real WeChat voice.

## Clips from videos (`labeling/sources/bili_ocr.py`, `bili_clips.py`, `unlabeled_clips.py`)

余姚话 videos whose on-screen text says what is spoken, downloaded to `data/bili/` (`<name>.mp4` + `.m4a`):
`bili_ocr.py` reads the text with macOS Vision (local) into timed segments, `bili_clips.py build` groups
them and cuts trimmed 16 kHz clips, `convert` has GPT (text only, no audio leaves the machine) propose the 余姚话 in 说法表
spellings from the on-screen text plus the local whisper-wu / wu-local drafts, and `publish`
writes `data/labels/extra_clips.jsonl`. The labeling page lists them under 视频 with the proposal as the
“字幕” draft and the Mandarin as the meaning; they are confirmed by ear like WeChat voices and are train-only.
Subtitles may trail the speech: the lag per range in `SOURCES` was found by matching subtitles against a
CTC character timeline of the whole audio.

`unlabeled_clips.py` does the same for downloaded 余姚话 / 慈溪话 audio without any on-screen text: it cuts
utterances by loudness, keeps the ones whose wu-ft draft reads as dialect (`build` → `publish`, page tab
网络), and cuts word-list readings one word per clip (`words`, page tab 词组). Shared audio helpers
(ffmpeg → 16 kHz, loudness, trimming) are in `sources/wavio.py`.
