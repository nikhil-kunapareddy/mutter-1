"""Word-piece tokenizer (SentencePiece unigram, 256 pieces).

Pieces are chunks like "▁the", "ing", "s". With 8x subsampling the model emits one prediction
per 80 ms, which is too coarse for single letters but fine for word-pieces.

CTC needs a special "blank" symbol meaning "nothing new here". We reserve id 0 for it and shift
every SentencePiece id up by one.
"""
from pathlib import Path

import sentencepiece as spm

BLANK = 0


class Tokenizer:
    def __init__(self, model_path: str | Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))

    def __len__(self) -> int:  # pieces + blank
        return self.sp.get_piece_size() + 1

    def encode(self, text: str) -> list[int]:
        return [i + 1 for i in self.sp.encode(text)]

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode([i - 1 for i in ids if i != BLANK])


def train_tokenizer(texts: list[str], out_prefix: Path, vocab_size: int = 256) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(texts),
        model_prefix=str(out_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1,   # no special tokens besides <unk>
        minloglevel=2,
    )
    return out_prefix.with_suffix(".model")
