"""Compact uncertainty-aware multi-task CNN for CRE activity prediction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from .sequence import normalize_sequence


MAX_SEQUENCE_LENGTH = 200


def encode_sequences(
    sequences: Iterable[str], max_length: int = MAX_SEQUENCE_LENGTH
) -> torch.Tensor:
    """Center-pad A/C/G/T one-hot channels and append an explicit validity mask."""
    sequences = [normalize_sequence(sequence) for sequence in sequences]
    encoded = np.zeros((len(sequences), 5, max_length), dtype=np.float32)
    base_to_channel = {base: channel for channel, base in enumerate("ACGT")}
    for row, sequence in enumerate(sequences):
        if len(sequence) > max_length:
            raise ValueError(f"sequence length {len(sequence)} exceeds {max_length}")
        invalid = set(sequence) - set("ACGT")
        if invalid:
            raise ValueError(f"invalid DNA alphabet: {sorted(invalid)}")
        start = (max_length - len(sequence)) // 2
        stop = start + len(sequence)
        encoded[row, 4, start:stop] = 1.0
        for column, base in enumerate(sequence, start=start):
            encoded[row, base_to_channel[base], column] = 1.0
    return torch.from_numpy(encoded)


def reverse_complement_tensor(inputs: torch.Tensor) -> torch.Tensor:
    """Reverse-complement an encoded batch while retaining the validity mask."""
    if inputs.ndim != 3 or inputs.shape[1] != 5:
        raise ValueError("expected tensor shape [batch, 5, length]")
    return inputs[:, [3, 2, 1, 0, 4], :].flip(dims=(-1,))


class HeteroscedasticCRECNN(nn.Module):
    """Predict three activity means and three data-dependent variance terms."""

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=7, padding=3),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.shared = nn.Sequential(
            nn.Linear(97, 96),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(96, 3)
        self.log_variance_head = nn.Linear(96, 3)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(inputs).squeeze(-1)
        length_fraction = inputs[:, 4, :].mean(dim=-1, keepdim=True)
        hidden = self.shared(torch.cat([hidden, length_fraction], dim=-1))
        mean = self.mean_head(hidden)
        raw_log_variance = self.log_variance_head(hidden).clamp(-8.0, 6.0)
        return mean, raw_log_variance


def heteroscedastic_nll(
    observed: torch.Tensor,
    mean: torch.Tensor,
    raw_log_variance: torch.Tensor,
    measurement_se: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL combining learned variance with reported MPRA standard error."""
    learned_variance = torch.nn.functional.softplus(raw_log_variance) + 1e-6
    total_variance = learned_variance + measurement_se.square()
    return 0.5 * torch.mean(
        (observed - mean).square() / total_variance + torch.log(total_variance)
    )
