# Speech-to-text experiments

What was measured on the way to the current setup (`combo` = local whisper-ft → deepseek-flash), newest
conclusions last. Provider names are those in `stt/core.py`; the cloud ones live in `stt/cloud/`. Fine-tuning
results are in `finetune/README.md`.

## First comparison (2026-11-14, 4 clips of ~2.5 s, all providers in parallel)

| clip | OpenAI (all 4 models) | DeepSeek | Volcengine (all 5 identical) |
|---|---|---|---|
| 余姚话 你在干嘛啦 | phonetic gibberish (侬嘞侬合适啦) | gibberish (侬来弄后戏啦) | **你在弄什么呢** (meaning right) |
| 余姚话 今天是礼拜天 | wrong (前面是黎巴捏) | **今日是礼拜日** | 起码是礼拜日 (half right) |
| 普通话 ×2 | correct | correct | correct |
| mean latency | 0.8–1.7 s | 1.4 s | flash 1.5 s, others 2.0–2.2 s |

Best single API: `volc-flash` (handles dialect, fastest Volcengine call), but it is not stable —
on reruns it sometimes returns 你在弄海鲜啦 for 你在干嘛啦.

Telling the model it's 余姚话 does not help (same 4 clips, 2026-11-14): OpenAI's `prompt` only fixes
the script (without it gpt-4o-transcribe returns pinyin/Tibetan/traditional characters), and a
detailed dialect description gives no better meaning; Volcengine `corpus.context` (dialog_ctx) changes
nothing on flash and made stream-2 worse once. Overrides stay available for experiments:
`stt.cloud.openai.transcribe(a, model, prompt=...)`, `stt.cloud.volc.flash/file/stream(..., extra={...})`.

## `combo` v1: several transcripts + an LLM that knows 余姚话 (stt/refine.py)

Dialect transcripts are either paraphrases (Volcengine) or homophone spellings of Wu sounds
(侬来弄后戏啦 = 侬来弄啥西啦 = 你在干什么啦). `combo` sends all of them to `deepseek-flash`
(thinking **disabled**) with a prompt saying the speaker may use 余姚话 and the transcripts are of one
sentence, and it returns one Mandarin sentence. Tested 2026-11-14:

| LLM step (same 3 candidates, 4 runs each) | correct meaning | latency |
|---|---|---|
| deepseek-flash, thinking disabled | 12/12 | 0.9–1.3 s |
| deepseek-flash, default (thinks) | 12/12 | 1–28 s (up to 13k chars of reasoning) |
| deepseek-v4-pro | correct | 17–70 s |
| gpt-5.4-mini / gpt-5.5, reasoning none | copies Volcengine's error (弄海鲜) | 0.6–1.4 s |
| gpt-6-luna / gpt-6-sol / gpt-6.1-sol / gpt-6.1-terra | copy 弄海鲜 0–2/4 right when Volcengine is wrong (fine otherwise) | 0.6–6 s |
| gpt-6-astra, reasoning low (no `none`) | 3/4 on the Volcengine-wrong case | 5–8 s |

Are all three transcripts needed? Same LLM step on every subset, 4 hard cases × 4 runs, counting
correct meaning: volc+ds+oa **16/16**, volc+ds 14/16 (fails half the time when Volcengine says
弄海鲜), volc+oa 10/16, ds+oa ~10/16, any single one 4–8/16. Volcengine gives the meaning, while
DeepSeek and OpenAI give two independent phonetic readings that out-vote a wrong Volcengine guess.
OpenAI is the fastest of the three (~0.7 s), so with parallel calls it adds no latency.

End to end on the 4 clips: `combo` 4/4 correct (你在干什么呢 / 今天是礼拜天 / 你在干什么 / 今天是礼拜天).
With `volc-flash` it averaged 2.4 s; since 2026-11-14 it uses `volc-file-2` (same transcripts, flash's
弄海鲜 error never seen on it, 1/5.6 the price) and averages 2.9 s (ASR ~1.9 s + LLM ~0.9 s). `Result.extra` holds each candidate and the latency split. If the
LLM call fails, `combo` falls back to the first transcript and records `refine_error`.

## Prices (checked 2026-11-14; USD at ~7.1 元)

| API | list price | ≈ 元 / hour of audio |
|---|---|---|
| DeepSeek ASR | 0.78 元/h, limited-time 50% off 0.39 元/h (no end date given) | 0.39–0.78 |
| 火山 录音文件识别 2.0 (`volc-file-2`) | 0.8 元/h | 0.8 |
| 火山 流式识别 2.0 (`volc-stream-2`) | 1 元/h | 1.0 |
| OpenAI gpt-4o-mini-transcribe | $0.003/min | 1.3 |
| OpenAI gpt-transcribe (`openai`) | $0.0045/min | 1.9 |
| 火山 录音文件识别 1.0 (`volc-file-1`) | 2.3 元/h (listed under 历史模型价格) | 2.3 |
| OpenAI gpt-4o-transcribe / whisper-1 | $0.006/min | 2.6 |
| 火山 极速版 (`volc-flash`), 流式 1.0 (`volc-stream-1`) | 4.5 元/h (历史模型价格) | 4.5 |

LLM step `deepseek-flash`: per 1M tokens 1 元 input (cache miss) / 4 元 output off-peak, double at peak
(weekdays 9–12, 14–18 Beijing); ~300 in + 15 out tokens per clip ≈ 0.0007 元.
`combo` per 3-second clip ≈ 0.004 元 (was ≈ 0.007 元 with volc-flash). Volcengine bills per second;
DeepSeek per second (rounded up); OpenAI minimum billing not stated. Sources:
https://developers.openai.com/api/docs/pricing, https://api-docs.deepseek.com/zh-cn/quick_start/pricing,
https://www.volcengine.com/docs/6561/1359370

## Long natural 余姚话 (2026-11-14, clips 16–20, 7–18 s, reference = the speaker's own Mandarin gloss)

Everything is mostly wrong here; the bottleneck is the speech APIs, not the LLM step.

| provider | mean CER | what it got |
|---|---|---|
| combo (file-2 + ds + oa) | 77% | gist of 19, part of 18 |
| volc-stream-1 / -2 | ~95% | the only ones to hear the key words of 18 and 20 |
| volc-file-1/-2, flash | 111–127% | heard clip 16 as English ("Yeah I you chewed it…") |
| deepseek / openai | ~90% | Wu homophones; nobody heard the key word of 17 |

Swapping the combo inputs (stream-2 instead of / in addition to file-2, or stream-2 + ds only) moved
mean CER only 84% → 79–82%, within run-to-run noise. On long clips the LLM also invents fluent detail
(e.g. a cat photo in 18), so treat combo output for long dialect speech as a gist, not a transcript.

### Alibaba / iFlytek dialect models on the same 9 clips (9–12 short, 16–20 long)

| provider | mean CER, all 9 | latency | notes |
|---|---|---|---|
| ali-audio31 (qwen-audio-3.1-asr-flash) | **56%** | 3.3 s | best single API on long clips (~70%): got the gist of 16, 19 and 20; short clips still wrong (你来弄海鲜啦, 礼拜日) |
| ali-fun (fun-asr-flash) | 67% | 2.2 s | |
| ali-qwen3 (qwen3-asr-flash) | 88% | 2.3 s | heard clip 16 as Thai |
| xf-dialect (讯飞 方言大模型) | 87% | 3.3 s | writes Wu characters (侬, 搿, 阿拉); misheard 17's key word; **non-deterministic** — same clip gives a full sentence, a fragment, or "" across runs; clip 16 as English |
| combo (unchanged) | 58–65% | | short clips right in meaning |

Feeding ali-audio31 and/or xf-dialect into combo's LLM step (3 runs each): 62–66% vs 58–65% for the
current inputs — no gain beyond noise. Extra Wu-ish candidates rescue clip 16 but nudge short clips
toward wrong word choices (礼拜日, 弄什么). combo is left as volc-file-2 + deepseek + openai.
Tencent not tested (no keys); xf-v2 (中英 V2.0) not enabled on the app.

### SiliconFlow and Gemini on the same 9 clips (2026-11-14)

| provider | mean CER | latency | notes |
|---|---|---|---|
| gemini-4 (gemini-4-carbon, prompt → Mandarin) | 58% (7/9; 2 timed out at 180 s) | 13 s | got 19's gist; elsewhere fluent but invented |
| sf-sensevoice | 74% | 4.3 s | Wu homophones, like deepseek |
| gemini-transcribe (gemini-3.5-transcribe) | 76% | 1.6 s | ignores prompts; empty output on 3 long clips, clip 16 as Cantonese |
| sf-telespeech | 78% | 3.0 s | "dialect" model; long clips drift into English fragments |
| gemini-pro (gemini-3.1-pro-preview) | 86% (8/9) | 41 s | confident invented stories (18, 19) |
| gemini-flash (gemini-3.8-flash) | 92% | 7.3 s | invented (a different food for 17's key word) |

None beats ali-audio31 (56%) or combo on these clips. Gemini models asked for "the Mandarin meaning" of
余姚话 produce fluent, plausible and wrong sentences — worse than a garbled transcript, because nothing
signals the error. gemini-omni-* only accepts the Interactions API and rejects audio input on this key.

### Local WenetSpeech-Wu model (`wu-local`, 2026-11-15)

Download (130 MB, from https://huggingface.co/csukuangfj2/sherpa-onnx-wenetspeech-wu-u2pp-conformer-ctc-zh-int8-2026-02-03):
`model.int8.onnx` + `tokens.txt` into `models/wu-u2pp-int8/`. Mean CER 68% (ali-audio31 59% in the same run),
but CER understates it: it writes **Wu characters** while the references are Mandarin glosses. By sound it
is the best ear tested — the only one to hear the key words of 17,
19 and 20, spelled as close Wu homophones. 0.02–0.17 s per clip, offline, free.

As a 4th combo input (3 LLM runs each): 66–73% vs 72% for the current set — no clear gain; deepseek-flash
does not turn such Wu-character homophones into Mandarin. The LLM also sometimes answers with
meta text ("这些识别结果都不太靠谱…请提供更准确的识别结果") instead of a sentence — a refine.py robustness bug.

### Local Whisper-Medium-Wu (`whisper-wu`, 2026-11-15)

`models/whisper-medium-wu/whisper-medium.pt` (3.3 GB WeNet fp32, from ASLP-lab/WenetSpeech-Wu-Speech-Understanding
`whisper/`) is mapped back to openai-whisper keys (`local_whisper_wu.convert()`, all keys match; the dropped
WeNet output-layer bias is ≤0.06) and saved as a 1.5 GB fp16 checkpoint. Decoded un-padded (as WeNet trained it),
language zh, beam 5. Reproduces the model card's test clips almost exactly.

On the 9 clips: mean CER 70% (wu-local 68%), ~1 s/clip on an M5 Max. Same kind of ear as wu-local — Wu
characters, phonetically close on the key words of 17–20 (18's had only been heard by volc-stream before).
Short clips no better
(9: 侬来侬欢喜啦, 10: 去末是礼拜日). Run with:
`uv run --with torch --with openai-whisper stt/cli.py -p whisper-wu …`


### Wu-aware LLM prompt (2026-11-15) — not adopted

Tried a prompt that labels each candidate (【吴语字】 for whisper-wu/wu-local, 【意译】 volc, 【同音字】
ds/oa), adds a general Wu glossary (侬 阿拉 伊 搿 呒没 葛末 侪 拨 物事 …; no clip-specific words) and tells
the LLM to trust the Wu transcripts' sounds. deepseek-flash, 3 runs × 9 clips:

| prompt / inputs | all | short 9–12 | long 16–20 |
|---|---|---|---|
| current prompt, file-2 + ds + oa | **59%** | **27%** | 84% |
| new prompt, same inputs | 64% | 34% | 87% |
| new prompt, + whisper-wu + wu-local | 63% | 37% | 83% |
| new prompt, file-2 + 2 Wu models | 64% | 45% | 80% |
| new prompt, 2 Wu models only | 64% | 44% | 80% |

The LLM still cannot map near-sound Wu spellings to real words (the key words of 17 and 18)
and copies wrong homophones; the Wu models' 起末/去嘛 (今朝) pull short clips off. combo stays as is.

Refusals: a "never refuse / never comment" rule in the prompt did not stop them (1 vs 0, 0 vs 1 over
two 6-run batches) and cost a few points on short clips, so `refine.py` now detects commentary
(`META` regex), retries once, then raises — `combo` falls back to the first transcript.

## Downstream check (2026-11-19)

deepseek-flash given the 5 WeChat + 11 语保 test clips, judged by hand against the labeled Mandarin meaning
(✓ right / △ gist / ✗ wrong), prompt with the 说法表 + `stt/yuyao_notes.md`:

| input | WeChat | 语保 |
|---|---|---|
| the exact 余姚话 label | 2/3/0 | 9/1/1 |
| whisper-ft | 1/0/4 | 7/2/2 |
| wu-ft | 0/0/5 | 4/1/6 |
| whisper-ft + wu-ft | 1/0/4 | 8/1/2 |

From a correct transcript the LLM gets the meaning; the bottleneck is recognition. A wrong key word is not
recovered but smoothed into a fluent wrong sentence (㑚好勿出去哉 → "你们好收衣服了"), so near-homophone
errors in content words and dropped negations are what matter, not CER as such.
