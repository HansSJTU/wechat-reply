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

from .audio import Audio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "models", "whisper-medium-wu")
WENET_CKPT = os.path.join(DIR, "whisper-medium.pt")
OPENAI_CKPT = os.path.join(DIR, "whisper-medium-wu.openai.pt")

DIMS = dict(n_mels=80, n_audio_ctx=1500, n_audio_state=1024, n_audio_head=16, n_audio_layer=24,
            n_vocab=51865, n_text_ctx=448, n_text_state=1024, n_text_head=16, n_text_layer=24)

_model = None


def ready() -> bool:
    if not (os.path.exists(OPENAI_CKPT) or os.path.exists(WENET_CKPT)):
        return False
    try:
        import torch  # noqa: F401
        import whisper  # noqa: F401
    except ImportError:
        return False
    return True


def _to_openai_name(name: str) -> str | None:
    if name.startswith("ctc.") or name == "decoder.output_layer.weight" or name == "decoder.output_layer.bias":
        return None  # CTC head and tied output layer: not part of openai-whisper
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


def convert() -> str:
    """WeNet checkpoint -> openai-whisper fp16 checkpoint (once)."""
    import torch
    sd = torch.load(WENET_CKPT, map_location="cpu", weights_only=False)
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
    torch.save({"dims": DIMS, "model_state_dict": out}, OPENAI_CKPT)
    return OPENAI_CKPT


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


def _load():
    global _model
    if _model is None:
        import torch
        import whisper
        if not os.path.exists(OPENAI_CKPT):
            convert()
        _model = whisper.load_model(OPENAI_CKPT, device="cpu")
        if torch.backends.mps.is_available():  # the sparse alignment_heads buffer can't move to MPS
            _model.register_buffer("alignment_heads", _model.alignment_heads.to_dense(), persistent=False)
            _model = _model.to("mps")
        _patch_encoder(_model)
    return _model


def transcribe(audio: Audio, beam_size: int = 5) -> str:
    import numpy as np  # heavy deps imported lazily so `import stt` works without them
    import torch
    import whisper
    model = _load()
    pcm = np.frombuffer(audio.pcm, dtype=np.int16).astype(np.float32) / 32768.0
    mel = whisper.log_mel_spectrogram(torch.from_numpy(pcm), n_mels=80).to(model.device)
    mel = mel[:, : mel.shape[-1] // 2 * 2]  # conv2 has stride 2
    opts = whisper.DecodingOptions(language="zh", task="transcribe", without_timestamps=True,
                                   beam_size=beam_size, fp16=model.device.type != "cpu")
    with torch.no_grad():
        return whisper.decode(model, mel, opts).text.strip()
