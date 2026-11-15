"""Live transcription from your microphone.

    uv run python -m mutter.live runs/conformer-s/best.pt

Talk, then pause. The dim line updates while you speak; after a short silence it's printed for
good and a new line starts. Ctrl+C quits.

The model isn't a streaming model (every step attends to the whole utterance), so the current
utterance is re-transcribed from its start every 0.4 s. At 8.4M parameters that's fast enough
on the CPU, which leaves the GPU free for a training run.

First run on macOS: allow your terminal app under System Settings > Privacy & Security >
Microphone. Without that permission the mic records pure silence.

The model only heard clean read audiobooks in training, so expect more mistakes on your mic than
its dev WER suggests: speak clearly, close to the mic, in a quiet room.

    uv run python -m mutter.live runs/conformer-s/best.pt --file clip.m4a   # replay a file as if live
"""
import argparse
import collections
import queue
import shutil
import sys
import time

import numpy as np
import torch

from .decode import greedy_decode, load_model
from .features import SAMPLE_RATE, LogMel
from .transcribe import load_audio

BLOCK = SAMPLE_RATE // 10            # audio arrives in 100 ms blocks
CALIBRATE = 5                        # first 0.5 s measures the background noise
DECODE_EVERY = 4                     # re-transcribe every 0.4 s while speaking
MAX_BLOCKS = 200                     # force a line break at 20 s (training clips are <= 25 s)
PREROLL = 3                          # keep 0.3 s before speech starts so the first sound isn't cut


def mic_blocks():
    import sounddevice as sd         # imported here so the rest of mutter doesn't need PortAudio
    q: queue.Queue = queue.Queue()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK,
                        callback=lambda data, *_: q.put(data[:, 0].copy())):
        while True:
            yield q.get()


def file_blocks(path):
    wav = load_audio(path)
    pad = np.zeros(SAMPLE_RATE, dtype=np.float32)               # silence before and after
    wav = np.concatenate([pad[:CALIBRATE * BLOCK], wav, pad])
    for i in range(0, len(wav), BLOCK):
        time.sleep(BLOCK / SAMPLE_RATE)                         # real-time pace, like a mic
        yield wav[i:i + BLOCK]


def level_db(block: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(block ** 2)) + 1e-10))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--file", help="replay this audio file in real time instead of using the mic")
    p.add_argument("--device", default="cpu", help="default cpu, so a training run keeps the GPU")
    p.add_argument("--silence", type=float, default=0.7, help="seconds of quiet that end a line")
    p.add_argument("--threshold-db", type=float,
                   help="fixed speech level in dBFS (default: 12 dB above the measured background)")
    args = p.parse_args()

    device = torch.device(args.device)
    model, tokenizer = load_model(args.checkpoint, device)
    featurizer = LogMel(model.cfg.n_mels)
    width = shutil.get_terminal_size().columns - 1

    def transcribe(blocks) -> str:
        feats = featurizer(torch.from_numpy(np.concatenate(blocks)))[None].to(device)
        logits, lens = model(feats, torch.tensor([feats.shape[1]], device=device))
        return greedy_decode(logits, lens, tokenizer)[0]

    def show(text: str, final: bool):
        if final:
            sys.stdout.write(f"\r\033[K{text}\n" if text else "\r\033[K")
        else:                                                   # dim, and only the tail if long
            sys.stdout.write(f"\r\033[K\033[2m{text[-width:]}\033[0m")
        sys.stdout.flush()

    print("Measuring background noise, stay quiet for half a second...")
    blocks = file_blocks(args.file) if args.file else mic_blocks()
    calib = [level_db(next(blocks)) for _ in range(CALIBRATE)]
    floor = float(np.median(calib))
    if floor < -150 and not args.file:
        sys.exit("The mic is sending pure silence. Allow your terminal app under System Settings > "
                 "Privacy & Security > Microphone, then restart the terminal.")
    end_blocks = max(1, round(args.silence * 10))
    threshold = lambda: args.threshold_db if args.threshold_db is not None else max(floor + 12, -55.0)
    print(f"Background {floor:.0f} dBFS, speech above {threshold():.0f} dBFS. "
          f"Listening: talk, pause to finish a line, Ctrl+C to quit.\n")

    utt: list[np.ndarray] = []
    preroll: collections.deque = collections.deque(maxlen=PREROLL)
    quiet = since_decode = 0
    try:
        for block in blocks:
            speech = level_db(block) > threshold()
            if not utt:                                         # waiting for speech to start
                if speech:
                    utt, quiet, since_decode = [*preroll, block], 0, 0
                else:
                    preroll.append(block)
                    floor = 0.9 * floor + 0.1 * max(level_db(block), -100.0)  # track the room
                continue
            utt.append(block)
            quiet = 0 if speech else quiet + 1
            since_decode += 1
            if quiet >= end_blocks or len(utt) >= MAX_BLOCKS:
                show(transcribe(utt), final=True)
                utt = []
                preroll.clear()
            elif since_decode >= DECODE_EVERY:
                show(transcribe(utt), final=False)
                since_decode = 0
    except KeyboardInterrupt:
        sys.stdout.write("\r\033[K")
    if utt:                                                     # Ctrl+C or file ended mid-sentence
        show(transcribe(utt), final=True)


if __name__ == "__main__":
    main()
