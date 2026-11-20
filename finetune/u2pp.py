#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "wenet @ https://github.com/wenet-e2e/wenet/archive/be6ad97fdb512589df113042f7ff3a29d392fb38.tar.gz",
#   "torch==2.5.1", "torchaudio==2.5.1", "numpy<2", "pyyaml", "onnx", "onnxruntime",
# ]
# ///
"""Fine-tune the WenetSpeech-Wu U2++ Conformer (the model behind wu-local) on the labeled WeChat voices.

  uv run stt/label.py export          data/labels/wenet/{train,test}.jsonl from the labeling page
  uv run finetune/u2pp.py eval        CER of the original model on test (and train)
  uv run finetune/u2pp.py train       fine-tune → models/wu-u2pp-ft/final.pt, then eval it
  uv run finetune/u2pp.py eval --ckpt models/wu-u2pp-ft/final.pt
  uv run finetune/u2pp.py export      final.pt → models/wu-u2pp-ft/model.int8.onnx + tokens.txt (sherpa-onnx,
                                      same format as wu-local; the labeling page's wu-ft draft uses it)

Starts from ASLP-lab's PyTorch checkpoint (models/wu-u2pp-torch/, the fp32 original of the int8 ONNX
wu-local runs). The vocabulary is fixed (5946 characters, same as wu-local's tokens.txt), so only
characters in it can be learned. WeNet needs torch 2.5 (newer torch breaks its imports); the recipe's
fbank/SpecAugment/speed-perturb front end is reimplemented here so no WeNet data pipeline is needed.
Trains on the Apple GPU (MPS) by default: CTC loss runs on CPU and the Conformer's depthwise conv is computed
from shifted copies (both have no / broken MPS kernels in torch 2.5), and batches are padded to a few fixed
shapes so compiled GPU kernels are reused: ~12 s per epoch for 11 min of audio vs ~38 s on CPU.
"""
import argparse
import json
import os
import random
import sys
import time
import wave
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CTC loss has no MPS kernel
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "models", "wu-u2pp-torch")
OUT = os.path.join(ROOT, "models", "wu-u2pp-ft")
DATA = os.path.join(ROOT, "data", "labels", "wenet")
RATE = 16000


# ---------------------------------------------------------------- model and data

def vocab() -> dict[str, int]:
    v = {}
    for line in open(os.path.join(BASE, "lang_char.txt"), encoding="utf-8"):
        ch, i = line.rsplit(maxsplit=1)
        v[ch] = int(i)
    return v


def load_model(ckpt: str):
    from wenet.utils.init_model import init_model
    configs = yaml.safe_load(open(os.path.join(BASE, "train.yaml"), encoding="utf-8"))
    configs["input_dim"] = 80
    configs["output_dim"] = len(vocab())
    model, configs = init_model(SimpleNamespace(checkpoint=ckpt), configs)
    return model


def read_wav(path: str) -> torch.Tensor:
    """16-bit mono WAV → float tensor (1, n) at 16 kHz, in int16 scale like WeNet."""
    with wave.open(path) as w:
        assert w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(1)
        rate = w.getframerate()
    x = torch.from_numpy(x).unsqueeze(0)
    return torchaudio.functional.resample(x, rate, RATE) if rate != RATE else x


def fbank(wav: torch.Tensor, train: bool) -> torch.Tensor:
    if train:  # speed perturb 0.9/1.0/1.1 as in the recipe
        speed = random.choice([0.9, 1.0, 1.1])
        if speed != 1.0:
            wav = torchaudio.functional.resample(wav, int(RATE * speed), RATE)
    f = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80, frame_length=25, frame_shift=10,
                                          dither=1.0 if train else 0.0, energy_floor=0.0,
                                          sample_frequency=RATE)
    if train:  # SpecAugment: 2 time masks ≤50 frames, 2 freq masks ≤10 bins
        f = f.clone()
        for _ in range(2):
            t = random.randint(1, min(50, max(1, f.shape[0] // 5)))
            s = random.randint(0, max(0, f.shape[0] - t))
            f[s:s + t] = 0
            b = random.randint(1, 10)
            s = random.randint(0, 80 - b)
            f[:, s:s + b] = 0
    return f


def load_split(split: str) -> list[dict]:
    path = os.path.join(DATA, f"{split}.jsonl")
    v = vocab()
    items = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        r["ids"] = [v.get(c, v["<unk>"]) for c in r["txt"]]
        r["audio"] = read_wav(r["wav"])
        items.append(r)
    return items


def batchify(items: list[dict], train: bool, bucket: bool = False) -> dict:
    """bucket: pad frames to a multiple of 128 and targets to a multiple of 16. On MPS every new tensor shape
    compiles new GPU kernels (~1 s); with random lengths that is nearly every step. Buckets keep the set of
    shapes small, so after the first epoch every step reuses compiled kernels. Lengths still mask the padding."""
    feats = [fbank(it["audio"], train) for it in items]
    flen = torch.tensor([f.shape[0] for f in feats])
    tlen = torch.tensor([len(it["ids"]) for it in items])
    feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    target = torch.nn.utils.rnn.pad_sequence([torch.tensor(it["ids"]) for it in items], batch_first=True,
                                             padding_value=-1)
    if bucket:
        up = lambda n, k: -(-n // k) * k
        feats = F.pad(feats, (0, 0, 0, up(feats.shape[1], 128) - feats.shape[1]))
        target = F.pad(target, (0, up(target.shape[1], 16) - target.shape[1]), value=-1)
    return {
        "feats": feats,
        "feats_lengths": flen,
        "target": target,
        "target_lengths": tlen,
        "langs": ["zh"] * len(items), "tasks": ["transcribe"] * len(items),
    }


# ---------------------------------------------------------------- eval

def edits(a: str, b: str) -> int:
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]


METHODS = ["ctc_greedy_search", "attention_rescoring"]


@torch.no_grad()
def evaluate(model, items: list[dict], show: bool = False) -> dict[str, float]:
    """CER per decoding method. CPU: decoding is small and some ops lack MPS kernels."""
    model.eval()
    model.to("cpu")
    chars = {m: 0 for m in METHODS}
    total = 0
    id2ch = {i: c for c, i in vocab().items()}
    for it in items:
        b = batchify([it], train=False)
        res = model.decode(METHODS, b["feats"], b["feats_lengths"], beam_size=10,
                           ctc_weight=0.3, reverse_weight=0.3)
        total += len(it["txt"])
        for m in METHODS:
            hyp = "".join(id2ch.get(t, "") for t in res[m][0].tokens)
            chars[m] += edits(hyp, it["txt"])
            if show and m == "attention_rescoring":
                print(f"  {it['key']}\n    标注 {it['txt']}\n    识别 {hyp}  [{edits(hyp, it['txt'])}/{len(it['txt'])}]")
    return {m: chars[m] / max(1, total) for m in METHODS}


def fmt(cer: dict) -> str:
    return "  ".join(f"{m.replace('_search', '')} {c:5.1%}" for m, c in cer.items())


def test_sets(te: list[dict]) -> dict[str, list[dict]]:
    """WeChat voices and the held-out 余姚 语保 sentences (yb_*) are reported apart: different speech."""
    sets = {"wechat": [x for x in te if not x["key"].startswith("yb_")],
            "语保": [x for x in te if x["key"].startswith("yb_")]}
    return {k: v for k, v in sets.items() if v}


def fmt_tests(model, te: list[dict]) -> str:
    return "   ".join(f"test[{k}] {fmt({m: c for m, c in evaluate(model, v).items() if m == 'ctc_greedy_search'})}"
                      for k, v in test_sets(te).items())


# ---------------------------------------------------------------- train

def ctc_loss_on_cpu(model):
    """Make the model trainable on MPS (torch 2.5). MPS has no CTC loss kernel: compute only the CTC loss on CPU,
    explicitly, with contiguous tensors (the .cpu() copy is differentiable); everything else stays on the GPU."""
    import types
    import torch.nn.functional as F

    def forward(self, hs_pad, hlens, ys_pad, ys_lens):
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        ys_hat = ys_hat.transpose(0, 1).log_softmax(2)
        loss = self.ctc_loss(ys_hat.cpu().contiguous(), ys_pad.cpu(), hlens.cpu(), ys_lens.cpu())
        loss = loss.to(hs_pad.device) / ys_hat.size(1)
        return loss, ys_hat.transpose(0, 1)

    model.ctc.forward = types.MethodType(forward, model.ctc)

    mps_safe_convs(model)


def mps_safe_convs(model):
    """Conv1d's backward pass fails on MPS in torch 2.5 ("view size is not compatible …"), for the Conformer's
    depthwise convs and Whisper's input convs alike. Compute every Conv1d with ops whose backward works:
    depthwise from shifted copies (y = Σ_k w[:, k] · x shifted by k), ordinary convs as unfold + matmul.
    Same results (checked to ~1e-6)."""
    import types
    import torch.nn.functional as F

    def depthwise(self, x):
        k, pad, dil = self.kernel_size[0], self.padding[0], self.dilation[0]
        xp = F.pad(x, (pad, pad))
        t = x.shape[2]
        y = sum(xp[:, :, i * dil:i * dil + t] * self.weight[:, 0, i].view(1, -1, 1) for i in range(k))
        return y + self.bias.view(1, -1, 1) if self.bias is not None else y

    def dense(self, x):
        k, pad, st, dil = self.kernel_size[0], self.padding[0], self.stride[0], self.dilation[0]
        xp = F.pad(x, (pad, pad))
        t = (xp.shape[2] - dil * (k - 1) - 1) // st + 1
        cols = torch.stack([xp[:, :, i * dil:i * dil + st * (t - 1) + 1:st] for i in range(k)], dim=3)
        y = cols.permute(0, 2, 1, 3).reshape(x.shape[0], t, -1) @ self.weight.reshape(self.out_channels, -1).t()
        if self.bias is not None:
            y = y + self.bias
        return y.transpose(1, 2)

    for mod in model.modules():
        if isinstance(mod, torch.nn.Conv1d) and isinstance(mod.padding, tuple):
            if mod.groups == mod.in_channels == mod.out_channels and mod.stride[0] == 1:
                mod.forward = types.MethodType(depthwise, mod)
            elif mod.groups == 1:
                mod.forward = types.MethodType(dense, mod)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tr, te = load_split("train"), load_split("test")
    secs = sum(it["audio"].shape[1] for it in tr) / RATE
    print(f"train {len(tr)} clips {secs / 60:.1f} min · test {len(te)} clips")
    model = load_model(os.path.join(BASE, "u2++.pt"))
    trs = random.Random(1).sample(tr, min(60, len(tr)))  # train CER on a fixed sample: CPU decoding is slow
    print(f"before   {fmt_tests(model, te)}   train {fmt(evaluate(model, trs))}")
    dev = torch.device(args.device)
    if dev.type == "mps":
        ctc_loss_on_cpu(model)
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params {sum(p.numel() for p in params) / 1e6:.1f} M on {dev}")
    # fused: one kernel per step instead of ~10 small ops (and a GPU sync) per parameter tensor
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, fused=dev.type == "mps")
    steps = args.epochs * ((len(tr) + args.batch - 1) // args.batch)
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    for ep in range(1, args.epochs + 1):
        model.to(dev)
        model.train()
        random.shuffle(tr)
        t0, losses = time.time(), []
        for i in range(0, len(tr), args.batch):
            out = model(batchify(tr[i:i + args.batch], train=True, bucket=dev.type == "mps"), dev)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            losses.append(out["loss"].item())
        line = f"epoch {ep:2d}  loss {np.mean(losses):7.2f}  {time.time() - t0:4.0f}s"
        if ep % args.eval_every == 0 or ep == args.epochs:
            line += f"   {fmt_tests(model, te)}   train {fmt(evaluate(model, trs))}"
        print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "final.pt")
    torch.save(model.state_dict(), path)
    print(f"saved {path}")
    print("test clips:")
    evaluate(model, te, show=True)


@torch.no_grad()
def export(ckpt: str):
    """Encoder + CTC to ONNX with sherpa-onnx's wenet_ctc metadata, then int8 (as sherpa-onnx's
    scripts/wenet/export-onnx.py does for the original wu-local model)."""
    import shutil
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    class OnnxModel(torch.nn.Module):
        def __init__(self, encoder, ctc):
            super().__init__()
            self.encoder, self.ctc = encoder, ctc

        def forward(self, x, x_lens):
            out, mask = self.encoder(x, x_lens, decoding_chunk_size=-1, num_decoding_left_chunks=-1)
            return self.ctc.log_softmax(out), mask.int().squeeze(1).sum(1)

    model = load_model(ckpt).eval()
    m = torch.jit.script(OnnxModel(model.encoder, model.ctc))
    fp32, int8 = os.path.join(OUT, "model.onnx"), os.path.join(OUT, "model.int8.onnx")
    x, x_lens = torch.rand(1, 1000, 80), torch.full((1,), 1000, dtype=torch.int64)
    torch.onnx.export(m, (x, x_lens), fp32, opset_version=13, input_names=["x", "x_lens"],
                      output_names=["log_probs", "log_probs_lens"],
                      dynamic_axes={"x": {0: "N", 1: "T"}, "x_lens": {0: "N"}, "log_probs": {0: "N", 1: "T"},
                                    "log_probs_lens": {0: "N"}})
    om = onnx.load(fp32)
    del om.metadata_props[:]
    for k, v in {"model_type": "wenet_ctc", "version": "1", "model_author": "ASLP-lab, fine-tuned locally",
                 "comment": "non-streaming", "subsampling_factor": model.encoder.embed.subsampling_rate,
                 "vocab_size": model.ctc.ctc_lo.weight.shape[0], "url": ckpt}.items():
        om.metadata_props.add(key=k, value=str(v))
    onnx.save(om, fp32)
    quantize_dynamic(model_input=fp32, model_output=int8, op_types_to_quantize=["MatMul"],
                     weight_type=QuantType.QInt8)
    os.remove(fp32)
    shutil.copy(os.path.join(BASE, "lang_char.txt"), os.path.join(OUT, "tokens.txt"))
    print(f"{int8} ({os.path.getsize(int8) / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", default=os.path.join(BASE, "u2++.pt"))
    x = sub.add_parser("export")
    x.add_argument("--ckpt", default=os.path.join(OUT, "final.pt"))
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--freeze-encoder", action="store_true", help="only train the CTC head and decoder")
    t.add_argument("--eval-every", type=int, default=2)
    t.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu",
                   help="mps (Apple GPU, ~3x faster than cpu here) or cpu")
    t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "eval":
        model = load_model(args.ckpt)
        te = load_split("test")
        for k, v in test_sets(te).items():
            print(f"test[{k}]  {fmt(evaluate(model, v, show=True))}")
        print(f"train {fmt(evaluate(model, load_split('train')))}")
    elif args.cmd == "export":
        export(args.ckpt)
    else:
        train(args)


if __name__ == "__main__":
    sys.exit(main())
