"""Transcribe your own audio files with a trained checkpoint.

    uv run python -m mutter.transcribe runs/conformer-s/best.pt my_recording.m4a

Any format ffmpeg can read works (m4a from Voice Memos, mp3, wav, ...). Audio is converted to
16 kHz mono, the rate the model was trained on. Works best on clips under ~30 seconds, since
training utterances are at most 25 s long.
"""
import argparse
import shutil
import subprocess

import numpy as np
import soundfile as sf
import torch

from .decode import greedy_decode, load_model
from .features import SAMPLE_RATE, LogMel
from .model import pick_device


def load_audio(path: str) -> np.ndarray:
    if shutil.which("ffmpeg"):
        raw = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
             "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
            capture_output=True, check=True,
        ).stdout
        return np.frombuffer(raw, dtype=np.float32).copy()
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"{path} is {sr} Hz; install ffmpeg (brew install ffmpeg) to resample")
    return wav.mean(axis=1)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("audio", nargs="+")
    args = p.parse_args()

    device = pick_device()
    model, tokenizer = load_model(args.checkpoint, device)
    featurizer = LogMel(model.cfg.n_mels)
    for path in args.audio:
        wav = load_audio(path)
        feats = featurizer(torch.from_numpy(wav))[None].to(device)
        logits, lens = model(feats, torch.tensor([feats.shape[1]], device=device))
        print(f"{path} ({len(wav) / SAMPLE_RATE:.1f}s): {greedy_decode(logits, lens, tokenizer)[0]}")


if __name__ == "__main__":
    main()
