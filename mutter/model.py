"""Conformer-CTC, written from scratch (Gulati et al., 2020, "Conformer: Convolution-augmented
Transformer for Speech Recognition").

Data flow for one batch (B utterances, T frames of 10 ms):

    log-mel (B, T, 80)
      -> ConvSubsampling       (B, T/8, 144)   3 stride-2 convs: one vector per 80 ms
      -> 16 x ConformerBlock   (B, T/8, 144)   each block: FF/2 -> self-attention -> conv -> FF/2
      -> LayerNorm + Linear    (B, T/8, 257)   a score for every word-piece + blank, per step

Self-attention gives every step a view of the whole utterance (global context); the depthwise
conv adds a sharp local view (~2.5 s window). Position is encoded with rotary embeddings (RoPE).

Pre-norm: every module normalizes its own input and adds its output to an untouched residual
stream; one LayerNorm sits before the head. The original Conformer also puts a LayerNorm at the
end of every block, but with 16 blocks, small batches and a short warmup that version never left
the CTC plateau here (it learned to erase the audio and output the word-piece prior), while a
2-block model left it within 1,000 steps. Without the per-block norm the audio has a direct path
to the head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class ConvSubsampling(nn.Module):
    """(B, T, n_mels) -> (B, T / 2**n_layers, d). Shrinks time so attention is cheap."""

    def __init__(self, n_mels: int, d: int, n_layers: int):
        super().__init__()
        layers, channels, freq = [], 1, n_mels
        for _ in range(n_layers):
            layers += [nn.Conv2d(channels, d, kernel_size=3, stride=2), nn.SiLU()]
            channels, freq = d, (freq - 3) // 2 + 1
        self.conv = nn.Sequential(*layers)
        self.n_layers = n_layers
        self.proj = nn.Linear(d * freq, d)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        x = self.conv(x.unsqueeze(1))                       # (B, d, T', F')
        b, c, t, f = x.shape
        x = self.proj(x.transpose(1, 2).reshape(b, t, c * f))
        for _ in range(self.n_layers):
            lens = (lens - 3) // 2 + 1                      # same arithmetic as each conv
        return x, lens


def rope_tables(length: int, head_dim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.arange(length, device=device).float()[:, None] * inv_freq[None]
    return angles.cos(), angles.sin()                       # each (T, head_dim / 2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of dims by a position-dependent angle, so q.k depends on relative distance."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class SelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads, self.dropout = n_heads, dropout
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, mask, cos, sin):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask[:, None, None, :],               # don't attend to padding
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class ConvModule(nn.Module):
    """pointwise conv + GLU -> depthwise conv -> BatchNorm -> SiLU -> pointwise conv."""

    def __init__(self, d: int, kernel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pointwise_in = nn.Conv1d(d, 2 * d, 1)
        self.depthwise = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.batch_norm = nn.BatchNorm1d(d)
        self.pointwise_out = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm(x).transpose(1, 2)                    # (B, d, T) for Conv1d
        x = F.glu(self.pointwise_in(x), dim=1)
        x = x.masked_fill(~mask[:, None, :], 0.0)           # padding must not leak into real frames
        x = F.silu(self.batch_norm(self.depthwise(x)))
        return self.drop(self.pointwise_out(x)).transpose(1, 2)


class FeedForward(nn.Sequential):
    def __init__(self, d: int, mult: int, dropout: float):
        super().__init__(
            nn.LayerNorm(d), nn.Linear(d, mult * d), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(mult * d, d), nn.Dropout(dropout),
        )


class ConformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.ff1 = FeedForward(d, cfg.ff_mult, cfg.dropout)
        self.attn_norm = nn.LayerNorm(d)
        self.attn = SelfAttention(d, cfg.n_heads, cfg.dropout)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.conv = ConvModule(d, cfg.conv_kernel, cfg.dropout)
        self.ff2 = FeedForward(d, cfg.ff_mult, cfg.dropout)
        # Every residual branch starts at exactly zero, so a fresh model is subsampling -> norm ->
        # head: it learns like a shallow model, and each block phases in as its output layer
        # grows. (Pre-norm alone was not enough for 16 blocks to leave the CTC plateau.)
        for last in (self.ff1[4], self.attn.out, self.conv.pointwise_out, self.ff2[4]):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x, mask, cos, sin):
        x = x + 0.5 * self.ff1(x)                           # "macaron" half-step feed-forwards
        x = x + self.attn_drop(self.attn(self.attn_norm(x), mask, cos, sin))
        x = x + self.conv(x, mask)
        return x + 0.5 * self.ff2(x)                        # no block-end norm (see module doc)


class ConformerCTC(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample = ConvSubsampling(cfg.n_mels, cfg.d_model, cfg.subsample_layers)
        self.in_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(ConformerBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, feats: torch.Tensor, lens: torch.Tensor):
        """feats (B, T, n_mels), lens (B,) -> logits (B, T', vocab), output lens (B,)."""
        x, lens = self.subsample(feats, lens)
        t = x.shape[1]
        mask = torch.arange(t, device=x.device)[None, :] < lens[:, None]   # True = real frame
        cos, sin = rope_tables(t, self.cfg.d_model // self.cfg.n_heads, x.device)
        x = self.in_drop(x)
        for block in self.blocks:
            x = block(x, mask, cos, sin)
        return self.head(self.final_norm(x)), lens


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
