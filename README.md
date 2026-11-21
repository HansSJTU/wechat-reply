# wechat-reply

Read WeChat 4.x (macOS) local messages and answer selected chats with an LLM.

## How it works (原理)

WeChat has no bot API on macOS, so the bot works from both ends of the desktop client: it reads
messages from WeChat's own local database as they are written, and sends replies through WeChat's
own internal send path.

```
 WeChat writes a message          ┌──────────── wxbot ──────────────┐
 to message_0.db-wal  ──kqueue──▶ │ 1 read    decrypt changed pages │
 (SQLCipher, encrypted)           │ 2 understand  text / voice→STT  │
                                  │               / image           │
                                  │ 3 reply   debounce → LLM        │
 WeChat sends it  ◀──Frida hook── │ 4 send    native send task      │
                                  └─────────────────────────────────┘
```

0. **Key (once).** The databases are SQLCipher 4 (AES-256-CBC, HMAC-SHA512 per 4 KB page). The
   key exists only in WeChat's memory, so it is captured once: an lldb script breaks on the
   CommonCrypto calls SQLCipher makes (`CCKeyDerivationPBKDF`, `CCCryptorCreate`, `CCHmacInit`)
   in a re-signed debuggable copy of WeChat, and `verify_keys.py` keeps the keys whose HMAC
   authenticates page 1 of a database. They go to `secrets/` and stay valid until re-login.
1. **Read.** New messages land in the database's `-wal` file first. `source.py` watches it with
   kqueue (no polling); on each write `livedb.py` decrypts only the pages whose ciphertext changed
   into a private in-memory SQLite copy, so plaintext never touches the disk. New rows of the
   `Msg_<md5(chat)>` tables become events (zstd-decompressed content, parsed XML for app messages).
2. **Understand.** Text is used as is. Voice is SILK in `media_0.db`, decoded to wav and sent to
   speech-to-text: `combo` asks several engines in parallel and lets an LLM merge their guesses,
   which is what makes dialect speech readable (`stt/README.md`). Images: the client itself only
   downloads the full size when one is opened, so the Frida agent starts that same CDN download
   (WeChat writes the decrypted JPEG to a file we name); otherwise the `.dat` files are decrypted
   (AES head + XOR tail, HEVC `wxgf` converted to JPEG with ffmpeg) and attached to the LLM call.
3. **Reply.** One thread per chat waits for a burst of messages to end (`debounce_seconds`),
   sends the last few hours of the conversation to the LLM, and gets one short reply. Rate limits,
   a stop file and a per-chat 开启 / 关闭 switch keep it in check.
4. **Send.** A Frida agent inside the debug copy of WeChat watches the client's own send routine.
   When you send any text by hand, it captures that call's context (the *seed*) and from then on
   builds new send tasks with it, so replies leave through WeChat itself like a normal message.
   Without a seed nothing can be sent, which is why one manual send is needed after each launch.

## Setup on a new Mac

Everything below runs on the Mac where WeChat is logged in. Keys, config and recorded messages
stay in the gitignored `secrets/`, `data/` and `config.local.toml`.

**Requirements.** Apple silicon Mac; WeChat 4.x from the official installer in `/Applications`.
Reading messages works across 4.x builds. **Sending works only on WeChat 4.2.0 build 270348
(4.2.0.12, arm64)**: the sender has that build's internal addresses built in, and checks the UUID of
`wechat.dylib` before touching anything (see "Other WeChat versions" below).

1. **Tools.** Xcode Command Line Tools (`lldb`, `codesign`, `cc`) and [uv](https://docs.astral.sh/uv/):
   ```bash
   xcode-select --install
   brew install uv
   git clone https://github.com/HansSJTU/wechat-reply.git && cd wechat-reply
   ```
   Python dependencies are declared inline in the scripts; `uv run` installs them on first use.
2. **Log in to WeChat** (`/Applications/WeChat.app`) and let it sync. To derive the image key later,
   at least one received image must be in a chat; open it once. Then **quit WeChat** (⌘Q).
3. **Make the debuggable copy.** `scripts/make_debug_wechat.sh` clones WeChat.app to
   `~/Applications/WeChat-debug.app` and ad-hoc re-signs it with `get-task-allow` and without hardened
   runtime, so lldb and Frida may attach. The copy has the same bundle id, so it shares the login and
   data of the original. Use only the copy from now on; rerun the script after every WeChat update.
4. **Capture the database key.** Start the hook first (it waits for WeChat to start), then launch the copy
   from a second terminal:
   ```bash
   mkdir -p -m 700 secrets
   lldb -o "command script import tools/hook_cc.py" -o "hookwait 180 $PWD/secrets/captures.tsv" -o quit
   ```
   ```bash
   open ~/Applications/WeChat-debug.app
   ```
   No sudo is needed; macOS may ask once to allow Developer Tools access. Open a few chats while it runs.
   It stops by itself once the account passphrase (the 256000-round PBKDF call SQLCipher makes when it
   opens a database) has been seen, and leaves WeChat running.
5. **Verify and store the keys**, then delete the raw captures:
   ```bash
   ACC=$(ls -d ~/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_*/ | head -1)
   uv run tools/verify_keys.py --account-dir "$ACC" --captures secrets/captures.tsv --out secrets/wechat_keys.json
   rm secrets/captures.tsv
   uv run tools/derive_image_key.py
   ```
   `verify_keys.py` must print `raw key: found`; otherwise quit WeChat and repeat step 4 (the passphrase
   only shows up while WeChat starts). The terminal may ask for access to other apps' data: allow it.
6. **API keys and config.**
   ```bash
   cp examples/stt_keys.toml secrets/stt_keys.toml && chmod 600 secrets/stt_keys.toml
   cp examples/config.local.toml config.local.toml
   uv run monitor.py chats            # usernames of your chats; put them in config.local.toml
   ```
   The defaults need OpenAI (replies) and DeepSeek (voice → Mandarin) keys (`examples/stt_keys.toml` explains each).
7. **Try it in the foreground.** With WeChat-debug running, start a dry run and send any text by hand
   from WeChat-debug (the sender's seed, needed once per WeChat launch):
   ```bash
   uv run monitor.py run --dry-run    # logs replies instead of sending them
   ```
   Drop `--dry-run` to really send.
8. **Run it in the background.** `scripts/install.sh` builds `~/Library/Application Support/wechat-monitor/
   WeChat Monitor.app` and loads it as a LaunchAgent. Add that app under System Settings → Privacy &
   Security → Full Disk Access (⌘⇧G to type the path) and allow its "access data from other apps"
   prompt; run `scripts/install.sh` again to restart it. Log: `data/monitor.log`, rotated past 5 MB to
   `.1`–`.3`; images in `data/media` expire after 120 days, voice clips are kept (`[housekeeping]`).

### Other WeChat versions

After an update, rerun steps 3–5 (and the key is re-captured the same way after a re-login). Reading,
transcription and LLM replies keep working as long as the database layout holds. Sending stops: the
sender refuses a `wechat.dylib` whose UUID differs from `EXPECTED_UUID` in `wxbot/sender/agent.js`. Porting it
means finding the same functions in the new build (the `offsets` table in `agent.js`; upstream ships
tables per version in `hook/versions/` of
[WeChat-ChatBot-Mac](https://github.com/Gusion1017/WeChat-ChatBot-Mac)) and updating the UUID
(the arm64 line of `dwarfdump --uuid ~/Applications/WeChat-debug.app/Contents/Resources/wechat.dylib`,
lower case without dashes). Until then, keep the
supported build installed and turn off WeChat's automatic updates.

## Reply bot (`wxbot/`, run by `monitor.py`)

A resident LaunchAgent that answers the chats in `config.toml` (your own chats, location and prompts go
in `config.local.toml`, gitignored, which overrides it key by key). Everything is driven by WeChat's
own database writes; nothing polls.

```
kqueue on message_*.db/-wal -> source.py -> pipeline.py (one thread per chat)
                                            understand.py: text / voice -> stt combo / image
                                            debounce a burst -> llm.py (OpenAI) -> sender/ (native)
```

- `source.py`: decrypts changed pages in memory (`livedb.py`), builds one event per new message
  and appends it to `data/messages.jsonl`. Read cursors live in `data/bot.db`, so a restart
  resumes; backlog older than `max_age_seconds` is recorded but not answered.
  Event fields: `chat`, `chat_name`, `is_group`, `sender`, `sender_name`, `is_self`, `time`, `type`
  (text/image/voice/app/emoji/…), `text`, `image_path`, `voice_path` (wav), `title`/`url`/`quote`
  for app messages, `latency_ms` (vs sender timestamp), `pipeline_ms` (WeChat DB write -> event).
  Every chat in the three lists of `config.toml` is watched and recorded the same way (voice saved as
  `data/media/<local_id>_<md5(chat)[:8]>_voice.wav`, picked up by `labeling/label.py`). The lists only set a
  chat's state before its first 开启 / 关闭: `chats` and `mention` start on, `listen` off. Group lines
  reach the LLM as `name：text`, with `[llm] group_prompt` appended to the system prompt.
- `pipeline.py`: ignores messages this account sends itself, waits `debounce_seconds` after a
  burst, sends the last `history_hours` of the conversation to the LLM and replies once. Limits:
  `max_per_minute` per chat; `touch data/STOP` pauses all replies. Each chat is switched by 开启 /
  关闭 — in a 1:1 chat the whole message, in a group after an @ of this account (its wxid in the
  message's `<atuserlist>`, or `@<nickname>`; @所有人 does not count):
  - 开启: voices are transcribed with `[stt] provider` (combo: local whisper-ft → Mandarin via deepseek-flash, wu-ft run and recorded alongside; the audio stays on this machine); every
    candidate and the Mandarin are appended to `data/voice_stt.jsonl` (the labeling page shows them as
    drafts) and the Mandarin enters the conversation as `[语音] …`. 1:1 chats are answered every time.
    Groups: an @ (or a quoted reply to the bot) is always answered; with `[group] auto` the bot may also
    speak up unasked — free rules first (no @-someone-else / emoji / links),
    then a `wait_seconds` pause so people answer each other, then a judge on the OpenAI Decisions API
    (`gpt-6-luna`: who is it for, what is it, and the probability of a few facts; it speaks up when every
    fact of one rule in `wxbot/llm.py` POLICY reaches `reply_threshold` or its `fact_thresholds`, and at
    least `voice_reply_threshold` when the new messages are all voices), then cooldowns; the reply
    model can still back out with `[不回]`. Every verdict goes to `data/group_decisions.jsonl`.
    The LLM's context starts at the 开启.
  - 关闭: record only — no cloud STT, no LLM, nothing leaves the machine. Voices are heard by the local
    models (whisper-ft and wu-ft) to catch a spoken 开启 (`[reply] command_stt` says whose text counts:
    whisper-ft, else wu-ft). Their text is kept on this machine in `data/voice_local.jsonl`, where the
    labeling page takes its drafts from; it never reaches the LLM.
  - Voice chat summaries (`[summary]`, groups that are 开启): once `min_voices` (5) transcribed voices
    have piled up since the last summary (or the last 开启) and nobody has sent a voice for `wait_seconds`
    (10 min; texts in between don't reset it), the bot posts 【聊天总结】 — the topics, who said what, what
    was decided. The timer runs apart from replies: @s and unprompted turns are handled as usual meanwhile. On demand: "@<me> 总结" or a voice "<name> 总结". A summary
    covers what came after the previous one, at most `max_hours` back.
  The switch can also be spoken: a voice that is just 开启 / 关闭 (in a group with the bot's name, e.g.
  "莫名其妙 开启").
  Both get a fixed confirmation; the state is kept in `bot.db` and survives restarts.
- `store.py`: `data/bot.db` holds the history the LLM sees. Native sends never appear in WeChat's
  own database, so the bot's replies are recorded only here.
- `llm.py`: placeholder chat bot on the OpenAI Responses API (`[llm]` in `config.toml`); the
  newest `max_images` images in the window are attached, the current date is added to the
  instructions, and the built-in `web_search` tool is on (`location` = where "today's weather" is).
  Markdown citations are stripped. At `reasoning_effort = "low"` a weather question got one empty
  weather-tool search and a made-up answer; `medium` falls back to a web search (~5 s).
- `sender/`: Frida agent inside `~/Applications/WeChat-debug.app` (4.2.0.12 arm64 only; the module
  UUID is checked). It needs a *seed*: one text sent by hand from the desktop client after each
  WeChat launch. One seed serves any number of sends (verified 2026-11-14 at 0 s, 60 s and 360 s,
  the last with the screen locked). Replies that arrive before a seed are held for
  `pending_ttl_seconds` while a macOS notification asks for the manual send. Sends are never retried.

```bash
uv run monitor.py run --dry-run          # foreground; log replies instead of sending
uv run monitor.py chats [query]          # find a chat's username
uv run monitor.py send --to <username> --text 'hi'
scripts/install.sh                       # (re)start the LaunchAgent; log: data/monitor.log
```

Run the debug copy (not `/Applications/WeChat.app`), logged in, and send any text from it once.
Images: the desktop client downloads the full image only when it is opened. For chats that can be
answered, the bot has WeChat-debug fetch it right away (`[media] fetch_full`, about 1 s; no seed
needed) and falls back to the thumbnail after `fetch_timeout_seconds`. WeChat answers -20003 while
the same image is being fetched already, which is the client's own thumbnail download for a moment
after the message row lands: the bot waits for the thumbnail first and retries on -20003, and uses
the client's full-size copy if one appears. WeChat's HEVC `wxgf` images are converted to JPEG with
ffmpeg (`brew install ffmpeg`).

The native sender is GPL-3.0-or-later; see `tools/NATIVE_SEND_NOTICE.md`.

## Background service

- Runs under launchd as `WeChat Monitor.app` (background-only, no UI). macOS only grants
  "data from other apps" to app bundles, so the bundle needs Full Disk Access. The grant is tied to
  the launcher's signature: `install.sh` rebuilds it only when `scripts/launcher.c` changes, and the
  launcher runs `monitor.py`, so keep that file name. `scripts/install.sh uninstall` removes it.
- The labeling page can run the same way: `scripts/install.sh label` installs `WeChat Label.app`
  (`com.local.wechat-label`, its own launcher `scripts/label_launcher.c`, so rebuilding one never resets
  the other's privacy grants). It keeps http://127.0.0.1:8765 up across logins and restarts it if it
  dies; it needs only the Documents folder. Log: `data/label.log`. `scripts/install.sh label uninstall`
  removes it.

## Repository layout

| dir | what | docs |
|---|---|---|
| `monitor.py`, `wxbot/` | the reply bot (read → understand → reply → send) | this file |
| `stt/` | speech-to-text: local Wu models + the LLM step behind `combo`; cloud APIs in `stt/cloud/` for comparison (`uv run stt/cli.py`) | `stt/README.md` |
| `labeling/` | labeling page (+ chat viewer) and builders of extra training clips | `labeling/README.md` |
| `finetune/` | fine-tuning and evaluating the local Wu models | `finetune/README.md` |
| `tools/`, `scripts/` | key capture, debuggable WeChat copy, LaunchAgent launchers | this file |
| `tests/` | offline unit tests (`uv run tests/run.py`) | — |
| `docs/` | measurements behind the current STT choices | `docs/stt-experiments.md` |

API keys for all of them: `secrets/stt_keys.toml` (template `examples/stt_keys.toml`).

## Keys (`secrets/wechat_keys.json`, mode 600 — never commit or share)

- `raw_key`: the account's SQLCipher passphrase (PBKDF2-SHA512, 256000 rounds; `kdf` holds the
  parameters) — opens every DB, including ones created later (`wxcrypto.key_for_db` derives and caches).
- `databases`: per-DB page keys, verified by page-1 HMAC.
- `image`: V2 `.dat` key, derived offline (`aes = md5(uin + wxid)[:16]`, `xor = uin & 0xff`).

## Re-capturing the DB key (only if it stops working, e.g. after re-login or a WeChat update)

Quit WeChat, then steps 3–5 of "Setup on a new Mac": rebuild the debug copy if WeChat updated, run the
lldb hook while the copy starts, `verify_keys.py` (keeps keys already in the file), `derive_image_key.py`.

## Formats

- DBs: SQLCipher 4, AES-256-CBC, 4096-byte pages, 80 reserved (IV 16 + HMAC-SHA512 64).
  Recent writes live in `-wal`; `wxcrypto.decrypt_db` merges committed WAL frames.
- `message_0.db` → `Msg_<md5(username)>` tables; `message_content` may be zstd
  (`WCDB_CT_message_content = 4`). Types: 1 text, 3 image, 34 voice, 43 video, 47 emoji, 49 app/link/file.
- Voice: `media_0.db` → `VoiceInfo.voice_data`, `0x02` + SILK v3 (strip first byte, decode with `pilk`, 24 kHz).
- Images: `msg/attach/<md5(chat)>/<YYYY-MM>/Img/<md5>[_t|_h].dat`; full-size only after download.
  The decrypted full size is usually `wxgf`: a header (its length in byte 4), then one HEVC stream,
  or two (alpha mask + picture, the larger one is the picture) each starting at a VPS NAL unit.
  `<img cdnmidimgurl aeskey>` in the message XML is what the CDN download needs.

## License

GPL-3.0-or-later, see [LICENSE](LICENSE). The native sender is adapted from GPL-3.0 code; its
provenance is in `tools/NATIVE_SEND_NOTICE.md`.
