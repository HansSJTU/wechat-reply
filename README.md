# wechat-reply

Read WeChat 4.x (macOS) local messages and answer selected chats with an LLM.

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
  Three lists in `config.toml`, all recorded the same way (voice saved as
  `data/media/<local_id>_<md5(chat)[:8]>_voice.wav`, picked up by `stt/label.py`): `chats` are always
  answered; `mention` chats are answered only when a text @-mentions this account (its wxid in the
  message's `source` `<atuserlist>`, or `@<nickname>` in the text; @所有人 does not count), and their
  voices never go to cloud STT (the LLM sees `[语音]`); `listen` chats never reach the pipeline.
  A group whose voices you want as dialect training data fits `mention`. Group lines reach the LLM as
  `name：text`, with `[llm] group_prompt` appended to the system prompt.
- `pipeline.py`: ignores messages this account sends itself, waits `debounce_seconds` after a
  burst, sends the last `history_hours` of the conversation to the LLM and replies once. Limits:
  `max_per_minute` per chat; `touch data/STOP` pauses all replies. In a group, "@<this account> 关闭"
  mutes that chat for `mute_hours` (24): still recorded, no LLM call; "@… 打开" resumes early and
  starts the LLM's context over (nothing from before the 打开); when the mute runs out on its own,
  the context starts at its end. Both get a fixed confirmation; the
  mute and the context start are kept in `bot.db`, so they survive restarts.
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
Images: the desktop client downloads the full image only when it is opened, so the LLM usually
gets the thumbnail; WeChat's `wxgf` format is not viewable and is passed as text.

The native sender is GPL-3.0-or-later; see `tools/NATIVE_SEND_NOTICE.md`.

## Background service

- Runs under launchd as `WeChat Monitor.app` (background-only, no UI). macOS only grants
  "data from other apps" to app bundles, so the bundle needs Full Disk Access. The grant is tied to
  the launcher's signature: `install.sh` rebuilds it only when `scripts/launcher.c` changes, and the
  launcher runs `monitor.py`, so keep that file name. `scripts/install.sh uninstall` removes it.

## Speech-to-text (`stt/`)

`uv run stt/cli.py` transcribes `data/media/*voice.wav` with every configured provider (OpenAI,
DeepSeek, five Volcengine variants) side by side. Keys go in `secrets/stt_keys.toml`. See `stt/README.md`.

Labeling for fine-tuning a local Wu model: `uv run stt/label.py serve` (see `stt/README.md`).

## Keys (`secrets/wechat_keys.json`, mode 600 — never commit or share)

- `raw_key`: the account's SQLCipher passphrase (PBKDF2-SHA512, 256000 rounds) — opens every DB,
  including ones created later (`wxcrypto.key_for_db` derives and caches).
- `databases`: per-DB page keys, verified by page-1 HMAC.
- `image`: V2 `.dat` key, derived offline (`aes = md5(uin + wxid)[:16]`, `xor = uin & 0xff`).

## Re-capturing the DB key (only if it stops working, e.g. after re-login)

1. Quit WeChat. `~/Applications/WeChat-debug.app` is an APFS clone of WeChat.app, ad-hoc
   re-signed with `get-task-allow` and without hardened runtime. If WeChat updated, rebuild it:
   `cp -cR /Applications/WeChat.app ~/Applications/WeChat-debug.app`, then
   `codesign --force --sign - --entitlements ent.plist ~/Applications/WeChat-debug.app`
   (`ent.plist` = original entitlements minus `com.apple.application-identifier`, plus
   `com.apple.security.get-task-allow`).
2. Start the hook, then launch the copy (no sudo needed):
   `lldb -o "command script import tools/hook_cc.py" -o "hookwait 180 $PWD/secrets/captures.tsv" -o quit`
   and `open ~/Applications/WeChat-debug.app`. It stops on its own after the raw key shows up.
3. `python3 tools/verify_keys.py --account-dir <xwechat_files/wxid_…> --captures secrets/captures.tsv --out secrets/wechat_keys.json`
   (store `raw_key` too), delete `captures.tsv`, quit the copy, reopen `/Applications/WeChat.app`.

`uv run tools/derive_image_key.py` re-derives the image key.

## Formats

- DBs: SQLCipher 4, AES-256-CBC, 4096-byte pages, 80 reserved (IV 16 + HMAC-SHA512 64).
  Recent writes live in `-wal`; `wxcrypto.decrypt_db` merges committed WAL frames.
- `message_0.db` → `Msg_<md5(username)>` tables; `message_content` may be zstd
  (`WCDB_CT_message_content = 4`). Types: 1 text, 3 image, 34 voice, 43 video, 47 emoji, 49 app/link/file.
- Voice: `media_0.db` → `VoiceInfo.voice_data`, `0x02` + SILK v3 (strip first byte, decode with `pilk`, 24 kHz).
- Images: `msg/attach/<md5(chat)>/<YYYY-MM>/Img/<md5>[_t|_h].dat`; full-size only after download.
