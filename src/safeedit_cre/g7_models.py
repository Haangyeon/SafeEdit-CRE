"""G7 model architectures: ResidualDilatedCNN (reviewer) and MultiKernelCNN (sealed evaluator).

Both architectures are intentionally different from the pilot HeteroscedasticCRECNN
and from each other, satisfying the architectural-independence requirement.
"""

from __future__ import annotations

import torch
from torch import nn

from .cnn_model import encode_sequences, reverse_complement_tensor


class ResidualBlock1D(nn.Module):
    """Residual block with dilated 1D convolution for sequence data."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class ResidualDilatedCNN(nn.Module):
    """Reviewer architecture: residual/dilated 1D CNN with multi-scale receptive field.

    Uses progressively increasing dilations (1, 2, 4, 8) to capture both local
    and long-range sequence patterns, fundamentally different from the pilot
    3-block MaxPool architecture.
    """

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.block1 = ResidualBlock1D(64, kernel_size=7, dilation=1, dropout=dropout)
        self.block2 = ResidualBlock1D(64, kernel_size=7, dilation=2, dropout=dropout)
        self.transition1 = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=1),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.block3 = ResidualBlock1D(96, kernel_size=5, dilation=4, dropout=dropout)
        self.block4 = ResidualBlock1D(96, kernel_size=5, dilation=8, dropout=dropout)
        self.transition2 = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.shared = nn.Sequential(
            nn.Linear(129, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(128, 3)
        self.log_variance_head = nn.Linear(128, 3)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.stem(inputs)
        hidden = self.block1(hidden)
        hidden = self.block2(hidden)
        hidden = self.transition1(hidden)
        hidden = self.block3(hidden)
        hidden = self.block4(hidden)
        hidden = self.transition2(hidden).squeeze(-1)
        length_fraction = inputs[:, 4, :].mean(dim=-1, keepdim=True)
        hidden = self.shared(torch.cat([hidden, length_fraction], dim=-1))
        mean = self.mean_head(hidden)
        raw_log_variance = self.log_variance_head(hidden).clamp(-8.0, 6.0)
        return mean, raw_log_variance


class MultiKernelCNN(nn.Module):
    """Sealed evaluator architecture: parallel multi-kernel (inception-style) 1D CNN.

    Uses parallel convolutions with kernel sizes 3, 7, 15, and 21 to capture
    motifs at multiple scales simultaneously, then fuses them. Fundamentally
    different from both the pilot CNN (sequential blocks) and the reviewer
    (residual dilated blocks).
    """

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.branch_k3 = nn.Sequential(
            nn.Conv1d(5, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )
        self.branch_k7 = nn.Sequential(
            nn.Conv1d(5, 48, kernel_size=7, padding=3),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )
        self.branch_k15 = nn.Sequential(
            nn.Conv1d(5, 48, kernel_size=15, padding=7),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )
        self.branch_k21 = nn.Sequential(
            nn.Conv1d(5, 48, kernel_size=21, padding=10),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.depth_conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4, groups=128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 96, kernel_size=1),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.head_conv = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=5, padding=2),
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
        b3 = self.branch_k3(inputs)
        b7 = self.branch_k7(inputs)
        b15 = self.branch_k15(inputs)
        b21 = self.branch_k21(inputs)
        fused = torch.cat([b3, b7, b15, b21], dim=1)
        fused = self.fusion(fused)
        fused = self.depth_conv(fused)
        hidden = self.head_conv(fused).squeeze(-1)
        length_fraction = inputs[:, 4, :].mean(dim=-1, keepdim=True)
        hidden = self.shared(torch.cat([hidden, length_fraction], dim=-1))
        mean = self.mean_head(hidden)
        raw_log_variance = self.log_variance_head(hidden).clamp(-8.0, 6.0)
        return mean, raw_log_variance


ARCHITECTURES = {
    "ResidualDilatedCNN": ResidualDilatedCNN,
    "MultiKernelCNN": MultiKernelCNN,
}


def load_g7_model(architecture: str, checkpoint_path, device: str = "cpu"):
    """Load a G7 model from checkpoint."""
    import torch

    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ARCHITECTURES[architecture]()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint
