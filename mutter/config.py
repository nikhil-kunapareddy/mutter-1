"""All hyperparameters in one place. Every field can be overridden from the command line,
e.g. `--d-model 256` or `--max-hours 8` (see train.py)."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_mels: int = 80             # log-mel frequency bins per 10 ms frame
    d_model: int = 144           # width of every Conformer layer ("Conformer-S")
    n_heads: int = 4             # attention heads (d_model / n_heads = 36 dims each)
    ff_mult: int = 4             # feed-forward hidden size = ff_mult * d_model
    n_layers: int = 16           # number of Conformer blocks
    conv_kernel: int = 31        # depthwise conv kernel: 31 frames * 80 ms ~ 2.5 s of context
    subsample_layers: int = 3    # 3 stride-2 convs -> 8x fewer frames (80 ms per output step)
    dropout: float = 0.1
    vocab_size: int = 257        # 256 word-pieces + 1 CTC blank (set from the tokenizer)


@dataclass
class TrainConfig:
    data_dir: str = "data"
    train_splits: tuple = ("train-clean-100",)
    dev_split: str = "dev-clean"
    tokenizer: str = "data/tokenizer/sp256.model"
    run_dir: str = "runs/conformer-s"

    # Batching: utterances of similar length are grouped so that
    # (number of utterances) x (longest utterance) <= max_batch_seconds.
    max_batch_seconds: float = 240.0
    max_utt_seconds: float = 25.0

    # Schedule. Training stops at whichever comes first: `epochs` or `max_hours`.
    epochs: int = 100
    max_hours: float = 6.0
    max_steps: int = 0           # >0: stop after this many optimizer steps (smoke tests)

    # Optimizer: AdamW with linear warmup then inverse-sqrt decay ("Noam"). The decay does not
    # depend on the total run length, so a run can be resumed night after night seamlessly.
    peak_lr: float = 1e-3
    warmup_steps: int = 2500
    weight_decay: float = 1e-3
    grad_clip: float = 5.0

    num_workers: int = 4
    log_every: int = 100
    dev_limit: int = 0           # >0: evaluate on only this many dev utterances
    resume: bool = True          # continue from run_dir/last.pt if it exists
    overfit_steps: int = 0       # >0: pipeline check, memorize one batch instead of training
    seed: int = 0
