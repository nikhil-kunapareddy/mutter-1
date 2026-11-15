# CLAUDE.md

Guidance for working in this repo. Read README.md for the user-facing overview.

## What this is

`mutter` is an English speech-to-text model built from scratch in PyTorch: an 8.4M-parameter
Conformer-CTC trained on LibriSpeech train-clean-100 (100 h). No pretrained weights, no
torchaudio: features, SpecAugment, model, CTC training and decoding are all hand-written.
It's sized to train on one Apple Silicon MacBook (M2 Pro, 16 GB, `mps` device).

## Commands

Everything runs through `uv` (Python 3.11, torch 2.15). Run from the repo root.

```bash
uv sync                                      # install deps into .venv
scripts/download_librispeech.sh              # ~7.6 GB into data/LibriSpeech (resumable)
uv run python -m mutter.prepare              # manifests + SentencePiece tokenizer (~10 s)

# Pipeline check (~1 min): memorize one dev batch; loss -> ~0, hyp == ref
uv run python -m mutter.train --overfit-steps 300 --max-batch-seconds 60 --run-dir runs/overfit

scripts/train_overnight.sh                   # one 6-hour training batch (caffeinate + resume)
uv run python -m mutter.train --run-dir runs/x --max-steps 3000 --dev-limit 300   # short experiment

uv run python -m mutter.evaluate runs/conformer-s/best.pt                 # test-clean/test-other WER
uv run python -m mutter.transcribe runs/conformer-s/best.pt clip.m4a      # needs ffmpeg for non-16k audio
uv run python -m mutter.live runs/conformer-s/best.pt                    # live mic; --file x.wav replays a file
```

There is no test suite. The overfit check plus short `--max-steps` runs are the tests.

## Layout

| Path | Role |
|---|---|
| `mutter/config.py` | `ModelConfig` + `TrainConfig` dataclasses. **Every field is auto-exposed as a CLI flag** by `train.py` (`max_hours` -> `--max-hours`, bools get `--no-x`). Add a hyperparameter here, not as a new argparse line. |
| `mutter/features.py` | `LogMel` (80 bands, 10 ms hop, 25 ms window, per-utterance mean/std norm) and `spec_augment` (2 freq masks <= 27 bins, 10 time masks <= 5% each, masked to 0) |
| `mutter/tokenizer.py` | SentencePiece unigram, 256 pieces. **Id 0 is the CTC blank**; SentencePiece ids are shifted +1 (vocab 257). |
| `mutter/data.py` | JSONL manifests, `SpeechDataset` (features computed in DataLoader workers; SpecAugment only when `train=True`), `DurationBatchSampler` (length-sorted, `count x longest <= max_batch_seconds`, batch order shuffled per epoch) |
| `mutter/model.py` | `ConvSubsampling` (3 stride-2 Conv2d, 8x time reduction -> 80 ms/step) -> 16 `ConformerBlock`s (d=144, 4 heads, RoPE, conv kernel 31, macaron FF) -> `final_norm` -> linear head. **Pre-norm with zero-started residual branches** (see Status for why); don't add back a block-end LayerNorm. Checkpoints from before this change don't load. |
| `mutter/train.py` | Training loop, Noam LR schedule, time budget, checkpoints, resume, `overfit_check` |
| `mutter/decode.py` | Greedy CTC decode, WER (jiwer), `load_model` |
| `mutter/prepare.py` | Builds `data/manifests/*.jsonl` and `data/tokenizer/sp256.model` |
| `mutter/evaluate.py`, `mutter/transcribe.py` | Test-set WER; transcribe arbitrary audio |
| `mutter/live.py` | Live mic transcription via `sounddevice`. Not a streaming model, so it re-decodes the whole current utterance every 0.4 s (~150 ms on CPU for 20 s); energy-based VAD ends a line after 0.7 s quiet. Runs on CPU by default so it can share the machine with training. Needs macOS mic permission for the terminal app. |
| `scripts/train_overnight.sh` | Warns if <40% RAM free, then `caffeinate -ims uv run python -m mutter.train "$@"` |

`data/`, `runs/` and `*.pt` are gitignored.

## How training behaves

- **Run dir** (`runs/conformer-s/` by default) is self-contained: `train.log`, `metrics.jsonl`
  (one line per epoch), `last.pt` (resume point, saved every epoch, atomic write), `best.pt`
  (lowest dev WER), and a copy of `tokenizer.model`.
- **Resume is automatic**: rerunning the same command continues from `last.pt`, restoring
  model, optimizer and scheduler. The architecture comes from the checkpoint, so model flags
  are ignored on resume. `--no-resume` refuses to overwrite an existing run; use a new `--run-dir`.
- **Time budget** (`--max-hours`, default 6) is checked only at epoch boundaries: training stops
  before an epoch that would exceed it. Ctrl+C (SIGINT) is safe; the interrupted epoch is redone.
- **LR schedule** is Noam (linear warmup, then 1/sqrt(step)) and depends only on the step count,
  so several 6-hour batches behave like one long run.
- **CTC loss runs on the CPU** (`train.py:ctc_loss`) because MPS has no CTC kernel. Logits are
  moved over each step (~10 ms). `reduction="mean"` divides by target length, so loss is per token.
- `--max-steps` stops mid-epoch but still runs dev eval and saves; pair it with `--dev-limit`.

## Hardware constraints (M2 Pro, 16 GB)

- About 380x realtime: one epoch (1,555 steps) is ~15 min plus ~1 min dev eval. A 6-hour batch
  is ~21-23 epochs; 100 epochs is ~5 batches.
- The GPU shares the 16 GB of RAM. If memory spills to swap, training stalls, so don't run
  two trainings at once, and quit browsers/editors first.
- Lid must stay open and the charger plugged in; macOS sleeps on lid close even with caffeinate.
- A training process launched from a Claude Code session is a child of that session and dies
  with it. Claude Code also kills its background tasks when system memory runs low (training
  leaves ~11% free). For anything over a few minutes, the user should start the command in their
  own terminal and Claude should watch `runs/<name>/train.log`. Never run two trainings at once.

## Debugging lessons

- **The overfit check is necessary, not sufficient.** Six utterances of different lengths can be
  memorized from length and position cues alone, without using the audio. Passing it proves the
  plumbing works, not that the model learns acoustics.
- **Symptom of collapse:** loss flat around 4.7 (the unigram-prior plateau for 256 pieces), dev
  WER 98-100%, empty hypotheses. The model outputs the token prior and ignores the input.
- **Healthy first epoch** (current model, defaults): loss 4.82 at step 500, 4.28 at 1,000,
  3.71 at 1,500; dev WER ~79% after epoch 1 with word-like output. Loss above ~4.6 at
  step 1,000 means something regressed.
- **Decisive cheap test:** `--n-layers 2` learns within ~1,000 steps. If a change makes even
  the 2-layer model stick at 4.7, suspect data/features/loss; if only deep models stick,
  suspect optimization/architecture.
- **Plateau level depends on the vocabulary:** ~4.7 for sp256 (unigram entropy 4.96
  nats/token), ~5.55 for sp1024 (5.82). Below it = the model is using the audio.
- **Fast check for "is the model using the audio?":** run the checkpoint on real features and on
  `torch.zeros_like(features)`; compare log-probs. Identical outputs mean the input was erased.
  Tracing the relative difference after each block shows where the signal dies.
- **MPS vs CPU:** gradients agree to ~1e-4 relative. Biases right before a BatchNorm
  (`conv.depthwise.bias`) have a true gradient of 0, so their CPU/MPS "mismatch" is noise.

## Status (2026-11-14)

- **Current run:** `runs/conformer-s`, the fixed model, 1 epoch done (step 1,555), dev WER
  79.4% (on a 300-utterance dev subset). Resume it with `scripts/train_overnight.sh`.
- **Why the model changed:** the original architecture (LayerNorm closing every block, the
  Conformer paper's layout) collapsed: 22 epochs at peak LR 1e-3 / warmup 2,500, loss stuck at
  4.7 from step ~1,000, dev WER never below 98%. Its output was identical for real audio and
  silence; the audio signal shrank ~10x per block and was gone by block 7.
- **Experiments** (logs in `runs/archive/`; old checkpoints don't load with the current code):

  | Run | Change | Result |
  |---|---|---|
  | `conformer-s-postnorm-collapsed` | original, defaults | stuck at 4.7 for 34k steps |
  | `exp1-lr3e-4` | peak LR 3e-4 | stuck at 4.7 through 3,000 steps |
  | `exp2-vocab1024` | 1,024 pieces, LR 5e-4 | stuck at its 5.6 plateau through 4,000 steps |
  | `exp3-short8s` | clips <= 8 s only | stuck at 4.72 through 1,400 steps |
  | `exp4-2layers` | 2 blocks | learned: 3.99 at step 1,500, dev WER 84% after epoch 1 |
  | `exp5-prenorm` | no block-end norm + `final_norm` | 4.63 at 1,500, WER 99%: not enough |
  | `exp6-zeroinit` (now `conformer-s`) | pre-norm + zero-started branches | 3.71 at 1,500, WER 79% |

- Ruled out directly: MPS vs CPU gradients, DataLoader workers vs `dataset[i]`, audio/transcript
  pairing, log-mel features (rendered to an image), token density (0.52 tokens+repeats per output
  step for sp256, all clips feasible).
- `data/tokenizer/sp1024.model` exists from exp2 but is unused.
