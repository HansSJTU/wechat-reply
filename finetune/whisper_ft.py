#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "wenet @ https://github.com/wenet-e2e/wenet/archive/be6ad97fdb512589df113042f7ff3a29d392fb38.tar.gz",
#   "torch==2.5.1", "torchaudio==2.5.1", "numpy<2", "pyyaml", "openai-whisper", "librosa",
# ]
# ///
"""Fine-tune Whisper-Medium-Wu (the model behind whisper-wu) on the labeled clips, like finetune/u2pp.py.

  uv run finetune/whisper_ft.py eval                     CER of the original model on the test sets
  uv run finetune/whisper_ft.py train [--epochs 8]      → models/whisper-wu-ft/final.pt (WeNet checkpoint)
  uv run finetune/whisper_ft.py export                  → models/whisper-wu-ft/whisper-ft.openai.pt (fp16,
                                                          what the labeling page's whisper-ft draft loads)

Starts from ASLP-lab's WeNet checkpoint (models/whisper-medium-wu/whisper-medium.pt, 769 M parameters) with
its train.yaml. Same data, splits and augmentation as u2pp.py, but Whisper's log-mel features (no 30 s padding,
as WeNet trains it) and its BPE tokenizer; the decoder is trained on <|zh|><|transcribe|><|notimestamps|> text.
The CTC head is switched off (ctc_weight 0): openai-whisper doesn't use it, and a CTC loss over 51 865 tokens
would have to run on CPU. Trains on the Apple GPU (MPS) with fused AdamW and bucketed batch shapes.
"""
import argparse
import os
import random
import re
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u2pp  # noqa: E402  (data loading, test-set split; puts the project root on sys.path)
from stt.cer import edits  # noqa: E402
from stt.paths import MODELS  # noqa: E402

BASE = os.path.join(MODELS, "whisper-medium-wu")
OUT = os.path.join(MODELS, "whisper-wu-ft")
VOCAB = 51865
NEED_GB = 24  # roughly what a train run takes (GPU memory + the Python process); see the "gpu peak" log lines
LABEL_SERVICE = "com.local.wechat-label"
_tok = None


def _run(*cmd) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def free_memory_gb() -> float:
    """What macOS can hand out without compressing or swapping: RAM minus what Activity Monitor calls Memory
    Used (app memory = anonymous − purgeable pages, wired, compressed). File cache counts as free."""
    out = _run("vm_stat")
    page = int(re.search(r"page size of (\d+)", out).group(1))
    n = lambda k: int(re.search(rf"{k}:\s+(\d+)", out).group(1))
    used = n("Anonymous pages") - n("Pages purgeable") + n("Pages wired down") + n("Pages occupied by compressor")
    return (int(_run("sysctl", "-n", "hw.memsize")) - used * page) / 2**30


def swap_used_gb() -> float:
    m = re.search(r"used = ([\d.]+)M", _run("sysctl", "-n", "vm.swapusage"))
    return float(m.group(1)) / 1024 if m else 0.0


def memory_check(need_gb: float = NEED_GB):
    """Warn before training if it won't fit in RAM. macOS doesn't fail then, it swaps: the 11-19 run (whisper-ft
    v2) ran ~3.5 h with the GPU mostly idle, 34 GB of swap and ~3 TB written to the SSD."""
    free = free_memory_gb()
    print(f"memory: {free:.0f} GB free, training needs ~{need_gb:.0f} GB, swap in use {swap_used_gb():.1f} GB",
          flush=True)
    pids = _run("pgrep", "-f", "python.*labeling/label.py serve").split()
    if pids:
        print(f"  ! the labeling page is running (pid {', '.join(pids)}): its drafts keep two Whisper-Medium "
              f"models on the GPU and it has used ~27 GB. Stop it for the run:\n"
              f"      launchctl bootout gui/{os.getuid()}/{LABEL_SERVICE}\n"
              f"    and bring it back afterwards with scripts/install.sh label", flush=True)
    if free < need_gb:
        print(f"  ! only {free:.0f} GB free: this run will swap and crawl. Biggest processes:", flush=True)
        for line in _run("top", "-l", "1", "-o", "mem", "-n", "6", "-stats", "pid,mem,command").splitlines()[-6:]:
            print("     ", line.rstrip(), flush=True)


def cap_mps(fraction: float):
    """Hard limit for MPS allocations, as a fraction of Metal's recommended working set (~28 GB on a 36 GB
    machine). PyTorch's default (1.7×) lets MPS grow well past physical RAM and macOS swaps silently; under the
    cap the allocator first frees its cache, then fails with 'MPS backend out of memory'. 0 = no limit."""
    if fraction > 0:
        torch.mps.set_per_process_memory_fraction(fraction)
        print(f"MPS memory capped at {fraction * torch.mps.recommended_max_memory() / 2**30:.0f} GB "
              f"(--mem-fraction {fraction})", flush=True)


def tokenizer():
    global _tok
    if _tok is None:
        from whisper.tokenizer import get_tokenizer
        _tok = get_tokenizer(multilingual=True, num_languages=99)
    return _tok


def load_model(ckpt: str):
    from wenet.utils.init_model import init_model
    configs = yaml.safe_load(open(os.path.join(BASE, "train.yaml"), encoding="utf-8"))
    configs["input_dim"] = 80
    configs["output_dim"] = VOCAB
    configs["model_conf"]["ctc_weight"] = 0.0
    model, _ = init_model(SimpleNamespace(checkpoint=ckpt), configs)
    return model


def log_mel(wav: torch.Tensor, train: bool) -> torch.Tensor:
    """WeNet's compute_log_mel_spectrogram (openai-whisper's, without padding to 30 s). wav: int16 scale."""
    import librosa
    x = wav.squeeze(0) / 32768.0
    if train:  # speed perturb 0.9/1.0/1.1
        speed = random.choice([0.9, 1.0, 1.1])
        if speed != 1.0:
            import torchaudio
            x = torchaudio.functional.resample(x, int(u2pp.RATE * speed), u2pp.RATE)
    stft = torch.stft(x, 400, 160, window=torch.hann_window(400), return_complex=True)
    mag = stft[..., :-1].abs() ** 2
    filters = torch.from_numpy(librosa.filters.mel(sr=u2pp.RATE, n_fft=400, n_mels=80))
    spec = torch.clamp(filters @ mag, min=1e-10).log10()
    spec = torch.maximum(spec, spec.max() - 8.0)
    f = ((spec + 4.0) / 4.0).transpose(0, 1)
    if train:  # SpecAugment as in u2pp.py
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
    items = u2pp.load_split(split)
    for it in items:
        it["ids"] = tokenizer().encode(it["txt"])
    return items


def batchify(items, train: bool, bucket: bool = False) -> dict:
    feats = [log_mel(it["audio"], train) for it in items]
    flen = torch.tensor([f.shape[0] for f in feats])
    feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    target = torch.nn.utils.rnn.pad_sequence([torch.tensor(it["ids"]) for it in items], batch_first=True,
                                             padding_value=-1)
    if bucket:  # few distinct shapes → MPS reuses compiled kernels (see u2pp.batchify)
        up = lambda n, k: -(-n // k) * k
        feats = F.pad(feats, (0, 0, 0, up(feats.shape[1], 128) - feats.shape[1]))
        target = F.pad(target, (0, up(target.shape[1], 16) - target.shape[1]), value=-1)
    return {"feats": feats, "feats_lengths": flen, "target": target,
            "target_lengths": torch.tensor([len(it["ids"]) for it in items]),
            "langs": ["zh"] * len(items), "tasks": ["transcribe"] * len(items)}


@torch.no_grad()
def evaluate(model, items, dev, show=False) -> float:
    """Greedy attention decoding (beam 1), CER on Han characters."""
    import re
    han = re.compile(r"[^㐀-鿿]")
    model.eval()
    errs = total = 0
    for it in items:
        b = batchify([it], train=False)
        try:
            res = model.decode(["attention"], b["feats"].to(dev), b["feats_lengths"].to(dev), beam_size=1,
                               infos={"tasks": ["transcribe"], "langs": ["zh"]})
            ids = [t for t in res["attention"][0].tokens if t < tokenizer().eot]
            hyp = han.sub("", tokenizer().decode(ids))
        except AssertionError:  # looped past the decoder's 448 positions without an end token: all wrong
            print(f"  ({it['key']}: decoding looped, counted as all wrong)", flush=True)
            hyp = ""
        e = edits(hyp, it["txt"])
        errs += e
        total += len(it["txt"])
        if show:
            print(f"  {it['key']}\n    标注 {it['txt']}\n    识别 {hyp}  [{e}/{len(it['txt'])}]")
    model.train()
    return errs / max(1, total)


def fmt_tests(model, te, dev) -> str:
    return "   ".join(f"test[{k}] {evaluate(model, v, dev):5.1%}" for k, v in u2pp.test_sets(te).items())


def length_batches(items, size):
    """Batches of similar length in random order: little padding and few distinct shapes (short word clips
    and long sentences mixed at random give a new padded shape almost every step, and MPS keeps each)."""
    items = sorted(items, key=lambda it: it["audio"].shape[1] * random.uniform(0.9, 1.1))
    batches = [items[i:i + size] for i in range(0, len(items), size)]
    random.shuffle(batches)
    return batches


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tr, te = load_split("train"), load_split("test")
    dev = torch.device(args.device)
    if dev.type == "mps":
        memory_check()
        cap_mps(args.mem_fraction)
    swap0 = swap_used_gb()
    model = load_model(os.path.join(BASE, "whisper-medium.pt"))
    if dev.type == "mps":
        u2pp.mps_safe_convs(model)
    model = model.to(dev)
    print(f"train {len(tr)} clips · test {len(te)} clips · {sum(p.numel() for p in model.parameters()) / 1e6:.0f} M "
          f"params on {dev}", flush=True)
    print(f"before   {fmt_tests(model, te, dev)}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, fused=dev.type == "mps")
    steps = args.epochs * ((len(tr) + args.batch - 1) // args.batch)
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    model.train()
    for ep in range(1, args.epochs + 1):
        t0, losses, peak = time.time(), [], 0
        for batch in length_batches(tr, args.batch):
            out = model(batchify(batch, True, bucket=dev.type == "mps"), dev)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            losses.append(out["loss"].item())
            if dev.type == "mps":  # torch 2.5 has no peak counter for MPS: sample after every step
                peak = max(peak, torch.mps.driver_allocated_memory())
        line = f"epoch {ep:2d}  loss {np.mean(losses):7.2f}  {time.time() - t0:4.0f}s"
        if ep % args.eval_every == 0 or ep == args.epochs:
            line += f"   {fmt_tests(model, te, dev)}"
            os.makedirs(OUT, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(OUT, "last.pt"))  # survives a crash later on
        if dev.type == "mps":  # every new shape caches memory; without this it grows until the GPU runs out
            torch.mps.empty_cache()
            line += (f"   gpu peak {peak / 2**30:.1f} GB, after empty_cache "
                     f"{torch.mps.driver_allocated_memory() / 2**30:.1f} GB   swap {swap_used_gb() - swap0:+.1f} GB")
        print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "final.pt")
    torch.save(model.state_dict(), path)
    print(f"saved {path}")
    for k, v in u2pp.test_sets(te).items():
        print(f"test[{k}] clips:")
        evaluate(model, v, dev, show=True)


def export(ckpt: str):
    """WeNet checkpoint → openai-whisper fp16 (the same mapping stt/local_whisper_wu.py uses)."""
    from stt import local_whisper_wu
    dst = os.path.join(OUT, "whisper-ft.openai.pt")
    local_whisper_wu.convert(ckpt, dst)
    print(f"{dst} ({os.path.getsize(dst) / 1e9:.1f} GB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", default=os.path.join(BASE, "whisper-medium.pt"))
    x = sub.add_parser("export")
    x.add_argument("--ckpt", default=os.path.join(OUT, "final.pt"))
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--eval-every", type=int, default=2)
    t.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--mem-fraction", type=float, default=1.0,
                   help="MPS memory cap as a fraction of the recommended working set (~28 GB here); 0 = none")
    a = ap.parse_args()
    if a.cmd == "eval":
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = load_model(a.ckpt).to(dev)
        te = load_split("test")
        for k, v in u2pp.test_sets(te).items():
            print(f"test[{k}] {evaluate(model, v, dev, show=True):5.1%}")
    elif a.cmd == "export":
        export(a.ckpt)
    else:
        train(a)


if __name__ == "__main__":
    main()
