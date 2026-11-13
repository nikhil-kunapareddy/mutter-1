"""Dataset, length-bucketed batching, and padding.

Manifests are JSONL files (one utterance per line) written by `python -m mutter.prepare`:
    {"audio": "data/LibriSpeech/.../19-198-0001.flac", "duration": 14.2, "text": "and the ..."}
"""
import json
import random
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .features import SAMPLE_RATE, LogMel, spec_augment
from .tokenizer import Tokenizer


def read_manifest(path: str | Path, max_seconds: float = float("inf"), limit: int = 0) -> list[dict]:
    with open(path) as f:
        entries = [json.loads(line) for line in f]
    entries = [e for e in entries if e["duration"] <= max_seconds]
    return entries[:limit] if limit else entries


def manifest_path(data_dir: str, split: str) -> Path:
    return Path(data_dir) / "manifests" / f"{split}.jsonl"


class SpeechDataset(Dataset):
    """Returns (features (frames, n_mels), token ids, transcript) for one utterance.
    Features are computed on the fly in DataLoader worker processes."""

    def __init__(self, entries: list[dict], tokenizer: Tokenizer, train: bool, n_mels: int = 80):
        self.audio = [e["audio"] for e in entries]
        self.texts = [e["text"] for e in entries]
        self.durations = [e["duration"] for e in entries]
        # Tokenize up front so worker processes don't need the SentencePiece object.
        self.tokens = [tokenizer.encode(t) for t in self.texts]
        self.train = train
        self.featurizer = LogMel(n_mels)

    def __len__(self) -> int:
        return len(self.audio)

    def __getitem__(self, i: int):
        wav, sr = sf.read(self.audio[i], dtype="float32")
        assert sr == SAMPLE_RATE, f"{self.audio[i]}: expected {SAMPLE_RATE} Hz, got {sr}"
        feats = self.featurizer(torch.from_numpy(wav))
        if self.train:
            feats = spec_augment(feats)
        return feats, torch.tensor(self.tokens[i], dtype=torch.long), self.texts[i]


def collate(batch):
    feats, tokens, texts = zip(*batch)
    feat_lens = torch.tensor([f.shape[0] for f in feats])
    token_lens = torch.tensor([t.shape[0] for t in tokens])
    padded = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)   # (B, max_frames, n_mels)
    targets = torch.cat(tokens)              # CTC takes all targets concatenated into 1-D
    return padded, feat_lens, targets, token_lens, list(texts)


class DurationBatchSampler(Sampler):
    """Sorts utterances by length and packs them into batches whose padded size
    (count x longest) stays under max_seconds. Similar lengths -> almost no wasted padding.
    Batch order is reshuffled every epoch."""

    def __init__(self, durations: list[float], max_seconds: float, shuffle: bool, seed: int = 0):
        order = sorted(range(len(durations)), key=lambda i: durations[i])
        self.batches, current = [], []
        for i in order:  # ascending, so durations[i] is the longest in `current`
            if current and (len(current) + 1) * durations[i] > max_seconds:
                self.batches.append(current)
                current = []
            current.append(i)
        if current:
            self.batches.append(current)
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return len(self.batches)


def make_loader(dataset: SpeechDataset, max_seconds: float, shuffle: bool,
                num_workers: int, seed: int = 0) -> DataLoader:
    sampler = DurationBatchSampler(dataset.durations, max_seconds, shuffle, seed)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
