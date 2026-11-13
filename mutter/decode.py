"""Turning model outputs into text, measuring word error rate, and loading checkpoints."""
from pathlib import Path

import jiwer
import torch

from .config import ModelConfig
from .model import ConformerCTC
from .tokenizer import BLANK, Tokenizer


def greedy_decode(logits: torch.Tensor, lens: torch.Tensor, tokenizer: Tokenizer) -> list[str]:
    """CTC greedy decoding: take the best symbol per step, merge repeats, drop blanks.
    e.g. steps [_, the, the, _, _, cat, _] -> [the, cat]."""
    best = logits.argmax(-1).cpu()
    texts = []
    for seq, n in zip(best, lens.cpu()):
        ids = torch.unique_consecutive(seq[:n]).tolist()
        texts.append(tokenizer.decode([i for i in ids if i != BLANK]))
    return texts


@torch.no_grad()
def evaluate(model, loader, tokenizer, device) -> tuple[float, list[tuple[str, str]]]:
    """Returns (word error rate, list of (reference, hypothesis))."""
    model.eval()
    refs, hyps = [], []
    for feats, feat_lens, _, _, texts in loader:
        logits, out_lens = model(feats.to(device), feat_lens.to(device))
        hyps += greedy_decode(logits, out_lens, tokenizer)
        refs += texts
    model.train()
    return jiwer.wer(refs, hyps), list(zip(refs, hyps))


def load_model(checkpoint: str | Path, device) -> tuple[ConformerCTC, Tokenizer]:
    """Load a checkpoint saved by train.py; its tokenizer lives next to it in the run dir."""
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ConformerCTC(ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    tokenizer = Tokenizer(checkpoint.parent / "tokenizer.model")
    return model.to(device).eval(), tokenizer
