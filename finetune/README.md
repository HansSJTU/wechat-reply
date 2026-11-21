# finetune — the local Wu models on the labeled clips

## U2++ (`finetune/u2pp.py`)

```
uv run labeling/label.py export                 # refresh data/labels/wenet/{train,test}.jsonl
uv run finetune/u2pp.py eval               # CER of the original model
uv run finetune/u2pp.py train --lr 1e-4 --epochs 20   # → models/wu-u2pp-ft/final.pt
```

Starts from ASLP-lab's fp32 PyTorch U2++ (`models/wu-u2pp-torch/`, 525 MB, from
`ASLP-lab/WenetSpeech-Wu-Speech-Understanding/u2++`; same 5946-char vocabulary as wu-local) with WeNet
pinned to a commit and torch 2.5.1 (newer torch breaks WeNet's imports). Full fine-tune on the Apple GPU
(MPS, default): the CTC loss runs on CPU and the Conformer's depthwise conv is computed from shifted copies
(torch 2.5 has no / a broken MPS kernel for them), batches are padded to a few fixed shapes so compiled
kernels are reused, and AdamW is fused — ~12 s per epoch for 11 min of audio vs ~38 s with `--device cpu`.
fbank, speed perturb and SpecAugment as in the recipe. `uv run finetune/u2pp.py export` writes
`models/wu-u2pp-ft/model.int8.onnx` + `tokens.txt` in wu-local's sherpa-onnx format; it runs as the `wu-ft`
provider and the third draft on the labeling page.

Test sets, reported apart: WeChat voices (the fixed hash split, 5 clips) and ~20% of the 余姚 语保
sentences (11 clips, read speech). 慈溪 clips are train only. CER of the int8 models:

| model | train data | WeChat test | 语保 test |
|---|---|---|---|
| wu-local (original) | — | 69.1% | 32.9% |
| v1 | 28 WeChat, 2.4 min | 58.8% | — |
| v2 | + all 44 余姚 语保, 5.8 min | 55.9% | (trained on them) |
| v3 | 31 WeChat + 33 余姚 语保 + 43 慈溪 语保 + 5 慈溪 course, 10.9 min | 52.9% | 15.7% |

(lr 1e-4, 20 epochs; the test CER bottoms out around epoch 12–16.) Drafts of clips a model was trained on
look much better than they are (it memorized them); judge it on new clips.

2026-11-19, after labeling the 词组 videos (17.8 min: + 410 word clips, 4 web clips, 1 WeChat voice; the test
set gained one WeChat voice, 17 clips). CER of the deployed models (int8 sherpa / openai-whisper fp16 beam 5,
audio resampled to 16 kHz as the bot does) on that test set:

| model | WeChat test | 语保 test |
|---|---|---|
| wu-local (original) | 66.2% | 32.1% |
| wu-ft v3 | 46.6% | 16.4% |
| **wu-ft v5** (lr 1e-4, 20 epochs) | **43.9%** | **13.6%** |
| whisper-wu (original) | 51.4% | 26.4% |
| whisper-ft v1 (112 clips) | 45.9% | 20.0% |
| **whisper-ft v2** (527 clips, lr 1e-5, 8 epochs) | **46.6%** | **15.7%** |

The WeChat test is 6 clips / 148 characters, so a few points either way is noise; 语保 is the steadier signal.
One test voice (60) is nearly the same sentence as a train voice (61); without it the numbers move < 3 points.

## Whisper (`finetune/whisper_ft.py`)

```
uv run finetune/whisper_ft.py train        # → models/whisper-wu-ft/final.pt (~2 min per epoch on MPS)
uv run finetune/whisper_ft.py export       # → whisper-ft.openai.pt (+ .output_layer.pt) for the whisper-ft provider
```

Full fine-tune of Whisper-Medium-Wu (869 M params, ~23 GB of GPU memory on MPS). Batches group clips of similar
length and the MPS cache is emptied each epoch: with short word clips and long sentences mixed at random almost
every step had a new shape, MPS kept memory for each, and an epoch went from 1.5 min to an hour before running out
of memory. A test clip whose greedy decoding loops past the decoder's 448 positions counts as all wrong.

The openai-whisper conversion (`stt/local_whisper_wu.py`) keeps WeNet's output layer: openai-whisper ties the
logits to the token embedding with no bias, while WeNet's output layer has a bias and, once fine-tuned, its own
weight. It is saved next to the checkpoint (`*.output_layer.pt`) and patched into the decoder; dropping it cost
3–6 points CER. Decoding falls back to temperature sampling when the beam output loops (compression ratio > 2.4),
as `whisper.transcribe` does.

## Meaning check (`finetune/meaning_eval.py`)

CER counts every particle; what matters is the Mandarin the bot ends up with. `meaning_eval.py` runs each
test clip's transcript through `stt/refine.py` (the bot's deepseek-flash prompt, 说法表 + `yuyao_notes.md`
included) and has deepseek-flash judge the result against the labeled Mandarin (✓ same meaning / △ gist /
✗ wrong; particles, tone and wording ignored); score = ✓ 1, △ ½, ✗ 0. deepseek-flash is not deterministic,
so `--runs 5` refines every input five times and reports the mean and range.

```
uv run finetune/meaning_eval.py decode whisper-v3 deployed models/whisper-wu-ft/whisper-ft.openai.pt
uv run finetune/meaning_eval.py decode wu-v6 u2pp models/wu-u2pp-ft/final.pt
uv run finetune/meaning_eval.py judge label whisper-v3 whisper-v3+wu-v6 --runs 5 [--llm gpt-6.2-sol]
```

`label` is the human 余姚话 label (the ceiling); `A+B` gives the LLM both transcripts. `deployed` decodes an
openai-whisper file as the bot does (fp16, beam 5, loop fallback); `whisper` (greedy) loops on some clips
and understates Whisper.

2026-11-20, after reviewing the labels against the 说法表 (682 train clips, 20.9 min; same 17 test clips),
deployed decoding, 5 deepseek-flash runs:

| ASR → deepseek-flash | WeChat (6) | 语保 (11) | CER WeChat / 语保 |
|---|---|---|---|
| human label | 67% | 95% | 0 / 0 |
| whisper-ft v2 | 27% (25–33) | 67% (64–73) | 48.0% / 15.7% |
| **whisper-ft v3** (lr 1e-5, 8 epochs) | **35%** (33–42) | **83%** (77–86) | **43.9% / 13.6%** |
| whisper-ft v2 + wu-ft v5 | 25% (17–33) | 74% (73–77) | — |
| whisper-ft v3 + wu-ft v6 | 33% (33–33) | 81% (77–86) | — |

wu-ft v6 (lr 1e-4, 20 epochs): CER 41.9% / 11.4% (v5: 42.6% / 13.6%). With deployed decoding wu-ft adds
nothing as a second LLM input, so `COMBO_LLM` stays whisper-ft. gpt-6.2-sol as the refine model scored about
the same as deepseek-flash (within noise) at ~1.3 s vs ~1.1 s. Even the human label reaches only 67% on the
WeChat clips: some errors are the LLM's (大X勿X read as "very X" until it went into `yuyao_notes.md`).
Deployed: whisper-ft v3, wu-ft v6.
