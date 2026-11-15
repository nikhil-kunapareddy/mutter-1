# mutter

A speech-to-text model built from scratch: an 8.4M-parameter **Conformer-CTC** trained on
100 hours of LibriSpeech, sized to train overnight on an Apple Silicon MacBook (M2 Pro, 16 GB).

Every piece is written by hand in PyTorch: log-mel features, SpecAugment, the Conformer
(attention + convolution), CTC training, and greedy decoding. No pretrained weights.

## Quick start

```bash
uv sync                                   # install dependencies into .venv
scripts/download_librispeech.sh           # ~7.6 GB: train-clean-100 + dev/test sets
uv run python -m mutter.prepare           # manifests + 256-piece tokenizer (~1 min)

# 2-minute pipeline check: the model should memorize one batch (loss -> ~0, hyp == ref)
uv run python -m mutter.train --overfit-steps 300 --max-batch-seconds 60 --run-dir runs/overfit

scripts/train_overnight.sh                # train one 6-hour batch; stops by itself
                                          # health check: loss < 4.3 by step 1,000 and dev WER
                                          # < 100% after epoch 1; stuck at ~4.7 = not learning
scripts/train_overnight.sh                # run again for the next batch (resumes)

uv run python -m mutter.evaluate runs/conformer-s/best.pt          # WER on test-clean/test-other
uv run python -m mutter.transcribe runs/conformer-s/best.pt me.m4a # your own recording
```

Before a long run: quit browsers and editors (the GPU shares the 16 GB; when it spills to swap,
training stalls), plug in the charger, and keep the lid open.

## How it works

```
waveform 16 kHz
  -> log-mel spectrogram       80 bands every 10 ms                  features.py
  -> SpecAugment (train only)  random frequency/time masks           features.py
  -> ConvSubsampling           3 stride-2 convs: 8x fewer frames     model.py
  -> 16 Conformer blocks       d=144, 4 heads, RoPE, conv kernel 31  model.py
                               pre-norm, each block starts as identity
  -> linear head               256 word-pieces + blank, every 80 ms  model.py
  -> CTC loss / greedy decode  merge repeats, drop blanks            train.py, decode.py
```

Why pre-norm with zero-started blocks: the original Conformer ends every block with a
LayerNorm. With 16 blocks, ~16-utterance batches and a 2,500-step warmup that version never
learned (loss stuck at ~4.7, the word-piece prior; it erased the audio instead). Removing the
per-block norm and starting each block's output at zero makes a fresh model behave like a
shallow one, which learns right away; the blocks phase in as they train.

Why 8x subsampling with word-pieces: one output every 80 ms is too coarse for single letters
(~15/s in read speech) but fine for word-pieces (~5/s), and it halves the encoder's work.

| File | What it does |
|---|---|
| `mutter/config.py` | All hyperparameters; each is also a CLI flag (`--d-model 256`) |
| `mutter/features.py` | Mel filterbank, log-mel, SpecAugment |
| `mutter/tokenizer.py` | SentencePiece word-pieces; id 0 is the CTC blank |
| `mutter/data.py` | Manifests, length-bucketed batching, padding |
| `mutter/model.py` | Conformer-CTC |
| `mutter/train.py` | Training loop, time budget, checkpoints, resume, overfit check |
| `mutter/decode.py` | Greedy CTC decoding, WER, checkpoint loading |
| `mutter/evaluate.py` / `transcribe.py` | Test-set WER / transcribe any audio file |

## Training outputs (`runs/conformer-s/`)

- `train.log` — loss, learning rate, speed (x realtime) and epoch ETA every 100 steps; dev WER
  and sample transcripts after every epoch
- `metrics.jsonl` — one line per epoch
- `last.pt` — resume point (saved after every epoch); `best.pt` — lowest dev WER so far
- `tokenizer.model` — copy of the tokenizer, so the run directory is self-contained

Training stops at whichever comes first: `--epochs` (100) or `--max-hours` (6). The learning
rate schedule (warmup, then 1/sqrt decay) doesn't depend on run length, so stopping and
resuming over several batches behaves like one long run. Ctrl+C is safe: the interrupted epoch
restarts on resume.

## Speed (measured, M2 Pro 16 GB)

- Training: ~380x realtime on train-clean-100 -> ~16 min/epoch + ~1 min dev eval
- One 6-hour batch ≈ 21 epochs (1,555 steps each); all 100 epochs ≈ 5 batches (~28 h)
- Data prep: download ~20 min at 8 MB/s, `mutter.prepare` ~10 s

## Expected results

Rough, from published results for similar small CTC models; greedy decoding, no language model.

| Training | Dev/test-clean WER | test-other WER |
|---|---|---|
| ~28 epochs (~8 h, 1–2 batches) | ~12–15% | ~30% |
| ~60–80 epochs (~17–23 h, 3–4 batches) | ~10% | ~25% |
