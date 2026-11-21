"""Local Whisper-Medium-Wu (WenetSpeech-Wu, ASLP-lab): whisper-medium fine-tuned on ~8000 h of Wu.

The release is a WeNet checkpoint (models/whisper-medium-wu/whisper-medium.pt, 3.3 GB fp32). `convert()`
maps it back to openai-whisper's layout once (inverse of WeNet's
wenet/models/whisper/convert_whisper_to_wenet_config_and_ckpt.py) and saves an fp16 openai-whisper
checkpoint, which `transcribe()` decodes with openai-whisper.

WeNet trained it on un-padded log-mels (no 30 s padding), so the encoder is patched to accept any
length and the mel is passed un-padded, as in training.
Source: https://huggingface.co/ASLP-lab/WenetSpeech-Wu-Speech-Understanding (whisper/)
"""
import os
import threading

from .audio import Audio
from .paths import MODELS

DIR = os.path.join(MODELS, "whisper-medium-wu")
WENET_CKPT = os.path.join(DIR, "whisper-medium.pt")
OPENAI_CKPT = os.path.join(DIR, "whisper-medium-wu.openai.pt")
# The same model fine-tuned on the labeled clips (finetune/whisper_ft.py train + export)
FT_CKPT = os.path.join(MODELS, "whisper-wu-ft", "whisper-ft.openai.pt")

DIMS = dict(n_mels=80, n_audio_ctx=1500, n_audio_state=1024, n_audio_head=16, n_audio_layer=24,
            n_vocab=51865, n_text_ctx=448, n_text_state=1024, n_text_head=16, n_text_layer=24)
MAX_FRAMES = DIMS["n_audio_ctx"] * 2  # 3000 mel frames = 30.0 s (positional_embedding cap)
MAX_SAMPLES = MAX_FRAMES * 160        # 16 kHz, hop 160 (10 ms/frame)

_models = {}
_lock = threading.RLock()


def ready(ckpt: str = OPENAI_CKPT) -> bool:
    if not (os.path.exists(ckpt) or (ckpt == OPENAI_CKPT and os.path.exists(WENET_CKPT))):
        return False
    try:
        import torch  # noqa: F401
        import whisper  # noqa: F401
    except ImportError:
        return False
    return True


def _to_openai_name(name: str) -> str | None:
    if name.startswith("ctc.") or name.startswith("decoder.output_layer."):
        return None  # CTC head: unused; output layer: saved apart (see convert)
    pairs = [
        ("encoder.embed.conv.0", "encoder.conv1"), ("encoder.embed.conv.2", "encoder.conv2"),
        ("decoder.embed.0", "decoder.token_embedding"),
        ("encoder.encoders", "encoder.blocks"), ("decoder.decoders", "decoder.blocks"),
        (".src_attn.linear_q", ".cross_attn.query"), (".src_attn.linear_k", ".cross_attn.key"),
        (".src_attn.linear_v", ".cross_attn.value"), (".src_attn.linear_out", ".cross_attn.out"),
        (".self_attn.linear_q", ".attn.query"), (".self_attn.linear_k", ".attn.key"),
        (".self_attn.linear_v", ".attn.value"), (".self_attn.linear_out", ".attn.out"),
        ("feed_forward.w_1", "mlp.0"), ("feed_forward.w_2", "mlp.2"),
        ("encoder.after_norm", "encoder.ln_post"), ("decoder.after_norm", "decoder.ln"),
    ]
    for a, b in pairs:
        name = name.replace(a, b)
    if name.startswith("decoder."):
        name = name.replace(".norm1.", ".attn_ln.").replace(".norm2.", ".cross_attn_ln.").replace(".norm3.", ".mlp_ln.")
    else:
        name = name.replace(".norm1.", ".attn_ln.").replace(".norm2.", ".mlp_ln.")
    return name


def convert(src: str = WENET_CKPT, dst: str = OPENAI_CKPT) -> str:
    """WeNet checkpoint -> openai-whisper fp16 checkpoint (once)."""
    import torch
    sd = torch.load(src, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    out = {}
    for k, v in sd.items():
        if k == "decoder.embed.1.pe":
            out["decoder.positional_embedding"] = v.squeeze(0)
        elif k == "encoder.embed.pos_enc.pe":
            out["encoder.positional_embedding"] = v.squeeze(0)[:DIMS["n_audio_ctx"]]
        else:
            n = _to_openai_name(k)
            if n:
                out[n] = v
    out = {k: v.half() if v.is_floating_point() else v for k, v in out.items()}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # openai-whisper ties the output layer to the token embedding and has no bias; WeNet's has its own weight
    # (it drifts from the embedding when fine-tuned) and a bias. Dropping them cost ~15 points CER on the
    # fine-tuned model. Kept next to the checkpoint and patched in by _load.
    torch.save({"weight": sd["decoder.output_layer.weight"].half(), "bias": sd["decoder.output_layer.bias"].half()},
               _out_path(dst) + ".tmp")
    torch.save({"dims": DIMS, "model_state_dict": out}, dst + ".tmp")
    os.replace(_out_path(dst) + ".tmp", _out_path(dst))
    os.replace(dst + ".tmp", dst)
    return dst


def _out_path(ckpt: str) -> str:
    return ckpt[: -len(".pt")] + ".output_layer.pt"


def _patch_decoder(model, ckpt: str):
    """Use WeNet's output layer (weight + bias) for the logits instead of the tied token embedding."""
    import types

    import torch
    w = torch.load(_out_path(ckpt), map_location="cpu", weights_only=True)
    model.decoder.register_buffer("out_w", w["weight"], persistent=False)
    model.decoder.register_buffer("out_b", w["bias"], persistent=False)

    def forward(self, x, xa, kv_cache=None):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset: offset + x.shape[-1]]
        x = x.to(xa.dtype)
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
        x = self.ln(x)
        return (x @ self.out_w.to(x.dtype).T + self.out_b.to(x.dtype)).float()

    model.decoder.forward = types.MethodType(forward, model.decoder)


def _patch_encoder(model):
    """Let the audio encoder take any number of frames (WeNet trained without 30 s padding)."""
    import types

    import torch.nn.functional as F

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        return self.ln_post(x)

    model.encoder.forward = types.MethodType(forward, model.encoder)


def _load(ckpt: str = OPENAI_CKPT):
    with _lock:
        if ckpt not in _models:
            import torch
            import whisper
            if ckpt == OPENAI_CKPT and not os.path.exists(_out_path(OPENAI_CKPT)):  # also redoes pre-fix conversions
                convert()
            m = whisper.load_model(ckpt, device="cpu")
            _patch_decoder(m, ckpt)
            if torch.backends.mps.is_available():  # the sparse alignment_heads buffer can't move to MPS
                m.register_buffer("alignment_heads", m.alignment_heads.to_dense(), persistent=False)
                m = m.to("mps")
            _patch_encoder(m)
            _models[ckpt] = m
        return _models[ckpt]


def _split_pcm(pcm, max_samples: int = MAX_SAMPLES, slack: int = 5 * 16000):
    """Split audio longer than Whisper's 30 s window into near-equal chunks, each cut at the quietest
    100 ms within ±5 s of the even split point, so no chunk is a near-silent sliver (Whisper
    invents text such as 谢谢观看 for those)."""
    import numpy as np
    if len(pcm) <= max_samples:
        return [pcm]
    chunks, pos, hop = [], 0, 1600  # 100 ms at 16 kHz
    while len(pcm) - pos > max_samples:
        rest = len(pcm) - pos
        target = pos + rest // -(-rest // max_samples)  # rest / chunks still needed
        lo, hi = max(pos + hop, target - slack), min(pos + max_samples, target + slack)
        seg = pcm[lo:hi]
        n = len(seg) // hop
        if n > 0:
            energy = (seg[: n * hop].reshape(n, hop) ** 2).mean(axis=1)
            cut = lo + int(np.argmin(energy)) * hop + hop // 2
        else:
            cut = hi
        chunks.append(pcm[pos:cut])
        pos = cut
    if pos < len(pcm):
        chunks.append(pcm[pos:])
    return chunks


def _decode_chunk(model, pcm_chunk, beam_size: int) -> str:
    import torch
    import whisper
    if len(pcm_chunk) < 400:  # shorter than Whisper's 25 ms STFT window (reflect pad 200)
        return ""
    mel = whisper.log_mel_spectrogram(torch.from_numpy(pcm_chunk), n_mels=80).to(model.device)
    frames = min(mel.shape[-1], MAX_FRAMES) // 2 * 2  # conv2 has stride 2; cap at n_audio_ctx * 2
    if frames < 2:
        return ""
    mel = mel[:, :frames]
    base = dict(language="zh", task="transcribe", without_timestamps=True, fp16=model.device.type != "cpu")
    with torch.no_grad():
        r = whisper.decode(model, mel, whisper.DecodingOptions(beam_size=beam_size, **base))
        # a loop ("渠日渠日渠日…") compresses well: re-decode with sampling, as whisper.transcribe does
        for t in (0.2, 0.4, 0.6, 0.8):
            if r.compression_ratio <= 2.4:
                break
            torch.manual_seed(0)
            r = whisper.decode(model, mel, whisper.DecodingOptions(temperature=t, best_of=5, **base))
        return r.text.strip()


def transcribe(audio: Audio, beam_size: int = 5, ckpt: str = OPENAI_CKPT) -> str:
    import numpy as np  # heavy deps imported lazily so `import stt` works without them
    pcm = np.frombuffer(audio.pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if len(pcm) < 400:
        return ""
    chunks = _split_pcm(pcm)
    # whisper.decode installs per-call KV-cache hooks on the shared model and seeds torch globally
    with _lock:
        model = _load(ckpt)
        parts = [_decode_chunk(model, c, beam_size) for c in chunks]
    return "".join(p for p in parts if p)
