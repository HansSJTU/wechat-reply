# stt — speech-to-text, several providers behind one call

Built for WeChat voice clips (Mandarin and 余姚话 / Wu dialect). Every clip is first normalized
to 16 kHz mono 16-bit WAV with ffmpeg (afconvert fallback), so any input format works.

```bash
uv run stt/cli.py --list                     # providers, ✓ = key configured
uv run stt/cli.py                            # all data/media/*voice.wav × all configured providers
uv run stt/cli.py a.m4a -p whisper-ft,wu-ft,combo
uv run stt/cli.py -e "参考1" -e "参考2" a.wav b.wav   # CER per file + per-provider latency summary
```

Results also go to `data/stt_results.jsonl`. From Python (`combo` needs `httpx`; the cloud providers also
`websockets>=13`):

```python
from stt import transcribe
r = transcribe("data/media/2_voice.wav")   # Result(text, error, latency, …)
```

## Keys

`secrets/stt_keys.toml` (mode 600, never commit; template in `examples/stt_keys.toml`), or the env vars
listed in `stt/keys.py`. `combo` needs only `deepseek_api_key` (the LLM step); the local models need none.
For Volcengine: `VOLC_APP_ID` + `VOLC_ACCESS_TOKEN` (used when both are set; otherwise `VOLC_API_KEY`, the
new console's single key — not the console's "Secret Key").

## Providers

| name | API | notes |
|---|---|---|
| **`combo`** (default) | local `whisper-ft` (audio stays on this machine; `wu-ft` also runs and is recorded, but since 2026-11-19 is not shown to the LLM) → `deepseek-flash` LLM with the 余姚话 notes (`stt/yuyao_notes.md`) and the 说法表 in its prompt | best on 余姚话; ~2–3 s; until 2026-11-19 it also used the cloud `volc-file-2` + `deepseek` + `openai` (docs/stt-experiments.md) |
| `openai` | `gpt-transcribe` | current model; `languages=[zh]` + a prompt asking for Simplified Chinese |
| `openai-4o`, `-4o-mini`, `-whisper` | older models | shut down 2027-02-26; kept for comparison |
| `deepseek` | `POST /v1/tools/asr` | no model/language/prompt params; ≤120 s, ≤8 MiB; lists only `zh`/`yue` |
| `volc-stream-2` | WSS `sauc/bigmodel_nostream`, `volc.seedasr.sauc.duration` | 豆包流式语音识别模型2.0 |
| `volc-stream-1` | same, `volc.bigasr.sauc.duration` | 流式语音识别大模型 1.0 |
| `volc-file-2` | `auc/bigmodel/submit`+`query`, `volc.seedasr.auc` | 豆包录音文件识别模型2.0; polled every 0.2 s, ~1.85 s |
| `volc-file-1` | same, `volc.bigasr.auc` | 录音文件识别大模型 1.0 |
| `volc-flash` | `auc/bigmodel/recognize/flash`, `volc.bigasr.auc_turbo` | 极速版, one synchronous call, ~1.3 s |
| `ali-qwen3` | 百炼 `qwen3-asr-flash`, OpenAI-compatible chat completions | lists 吴语 |
| `ali-fun` | 百炼 `fun-asr-flash-2026-06-15`, multimodal-generation | lists 吴语 |
| `ali-audio31` | 百炼 `qwen-audio-3.1-asr-flash`, multimodal-generation | lists 吴语; outputs Mandarin for dialect by default |
| `tencent-dialect` | SentenceRecognition `16k_zh_dialect` (TC3 signing) | 23 dialects incl. 上海话; ≤60 s |
| `tencent-flash` | FlashRecognize `16k_zh_en` 大模型1.0 (needs AppID) | 中英粤 + 上海话 etc. |
| `tencent-2.0` | CreateRecTask `16k_zh_en_2.0` + DescribeTaskStatus | 31 dialects incl. 宁波话, 吴语 |
| `xf-v2` | 讯飞 语音听写大模型 V2.0, websocket `iat.cn-huabei-1.xf-yun.com/v1` | 202 dialects incl. 宁波话; ≤60 s |
| `xf-dialect` | 讯飞 方言大模型 (same websocket, `domain=slm`, `accent=mulacc`) | 普通话 + 202 dialects auto-detected; needs the 方言识别大模型 service with quota on the app (else code 11201) |
| `sf-sensevoice` | 硅基流动 FunAudioLLM/SenseVoiceSmall, `/v1/audio/transcriptions` | small multilingual model |
| `sf-telespeech` | 硅基流动 TeleAI/TeleSpeechASR | China Telecom, advertised for dialects |
| `gemini-flash` / `gemini-pro` | Gemini generateContent with inline WAV, prompt asks for the Mandarin meaning | general multimodal LLM: hears + translates in one call |
| `wu-local` | **local, offline**: WenetSpeech-Wu U2++ Conformer CTC, sherpa-onnx int8 (130 MB) in `models/wu-u2pp-int8/` | trained on ~8000 h of Wu; outputs Wu characters; ~0.1 s/clip on CPU |
| `whisper-wu` | **local**: Whisper-Medium-Wu (WenetSpeech-Wu), WeNet ckpt converted to openai-whisper on first use, MPS | needs `--with torch --with openai-whisper`; ~1 s/clip; Wu characters |

Each Volcengine variant needs its service enabled for the app in the console; an unenabled one
fails with `45000030`-style "grant/resource" errors while the others keep working.

Dialects: Volcengine's bigmodel family (no `language` set) documents Shanghainese (吴语) plus
Sichuan/Cantonese/Min Nan; none of the three vendors lists 余姚话, so the CLI comparison is the test.
`volc-file-*` sends base64 `audio.data`; the docs only show `audio.url`, so if that variant
rejects it, the submit API needs a public URL instead.

## Layout

| file | what |
|---|---|
| `core.py` | provider registry, `combo`, `transcribe()` |
| `audio.py` | any input → 16 kHz mono wav (`Audio`) |
| `local_wu.py`, `local_whisper_wu.py` | the local Wu models (sherpa-onnx U2++, openai-whisper) |
| `refine.py` + `yuyao_notes.md` | `combo`'s LLM step: transcripts → Mandarin, with the 余姚话 notes and 说法表 |
| `cloud/` | the cloud APIs, for comparison only; imported lazily, so the bot never loads them |
| `keys.py`, `paths.py`, `cer.py` | API keys, project paths, character error rate |
| `cli.py` | side-by-side comparison on any clips |

Labeling and dataset tools: `labeling/README.md`. Fine-tuning: `finetune/README.md`. Every measurement
behind the current choices (provider comparisons, prices, prompts, model versions): `docs/stt-experiments.md`.
