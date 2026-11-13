"""Audio -> log-mel spectrogram, plus SpecAugment. Written from scratch with torch.stft.

A log-mel spectrogram is how the model "sees" sound: every 10 ms we measure how much energy
the audio has in 80 frequency bands spaced like human hearing (the mel scale), then take the log.
"""
import math
import random

import torch

SAMPLE_RATE = 16000
HOP = 160        # 10 ms between frames
WIN = 400        # 25 ms analysis window
N_FFT = 512


def mel_filterbank(n_mels: int, n_fft: int = N_FFT, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Triangular filters that pool FFT bins into mel bands. Returns (n_mels, n_fft // 2 + 1)."""
    def hz_to_mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    mel_points = torch.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2)
    hz_points = 700.0 * (10 ** (mel_points / 2595.0) - 1.0)
    fft_freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    lower, center, upper = hz_points[:-2, None], hz_points[1:-1, None], hz_points[2:, None]
    rising = (fft_freqs - lower) / (center - lower)
    falling = (upper - fft_freqs) / (upper - center)
    return torch.clamp(torch.minimum(rising, falling), min=0.0)


class LogMel:
    """Callable: waveform (samples,) float32 in [-1, 1] -> normalized features (frames, n_mels)."""

    def __init__(self, n_mels: int = 80):
        self.window = torch.hann_window(WIN)
        self.filters = mel_filterbank(n_mels)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(wav, N_FFT, HOP, WIN, self.window, center=True, return_complex=True)
        power = spec.abs() ** 2                                  # (freq_bins, frames)
        logmel = torch.log(self.filters @ power + 1e-6).T        # (frames, n_mels)
        # Per-utterance normalization: removes loudness/microphone differences between recordings.
        return (logmel - logmel.mean(0)) / (logmel.std(0) + 1e-5)


def num_frames(num_samples: int) -> int:
    return 1 + num_samples // HOP


def spec_augment(x: torch.Tensor, freq_masks: int = 2, max_freq: int = 27,
                 time_masks: int = 10, max_time_frac: float = 0.05) -> torch.Tensor:
    """SpecAugment (Park et al., 2019): blank out random frequency bands and time spans.
    Forces the model to not rely on any single band or moment -> big accuracy gain on small data.
    Masks are set to 0, which is the mean after normalization."""
    x = x.clone()
    frames, bins = x.shape
    for _ in range(freq_masks):
        width = random.randint(0, max_freq)
        start = random.randint(0, bins - width)
        x[:, start:start + width] = 0.0
    max_width = int(max_time_frac * frames)
    for _ in range(time_masks):
        width = random.randint(0, max_width)
        start = random.randint(0, max(0, frames - width))
        x[start:start + width] = 0.0
    return x
