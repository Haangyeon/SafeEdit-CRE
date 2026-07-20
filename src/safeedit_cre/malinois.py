"""Minimal, dependency-light loader for the published Malinois predictor."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .sequence import normalize_sequence, reverse_complement


MPRA_UPSTREAM = (
    "ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCT"
    "CAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACAT"
    "CTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACT"
    "AGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG"
)
MPRA_DOWNSTREAM = (
    "CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCG"
    "GCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCA"
    "CCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGC"
    "CACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT"
)


def pad_mpra_sequence(sequence: str, padded_length: int = 600) -> str:
    sequence = normalize_sequence(sequence)
    padding = padded_length - len(sequence)
    if padding < 0:
        raise ValueError(f"sequence length {len(sequence)} exceeds {padded_length}")
    upstream_length = padding // 2
    downstream_length = padding - upstream_length
    if upstream_length > len(MPRA_UPSTREAM) or downstream_length > len(MPRA_DOWNSTREAM):
        raise ValueError("published MPRA flanks are too short for requested padding")
    return MPRA_UPSTREAM[-upstream_length:] + sequence + MPRA_DOWNSTREAM[:downstream_length]


def one_hot(sequences: Iterable[str]) -> torch.Tensor:
    sequences = list(sequences)
    if not sequences:
        return torch.empty((0, 4, 0), dtype=torch.float32)
    length = len(sequences[0])
    if any(len(sequence) != length for sequence in sequences):
        raise ValueError("one-hot input sequences must have equal length")
    encoded = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8).reshape(
        len(sequences), length
    )
    lookup = np.full(256, -1, dtype=np.int8)
    lookup[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4, dtype=np.int8)
    indices = lookup[encoded]
    if np.any(indices < 0):
        raise ValueError("one-hot input contains a non-ACGT base")
    array = np.eye(4, dtype=np.float32)[indices].transpose(0, 2, 1)
    return torch.from_numpy(array.copy())


class Conv1dNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.bn_layer = nn.BatchNorm1d(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.bn_layer(self.conv(inputs))


class LinearNorm(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn_layer = nn.BatchNorm1d(out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.bn_layer(self.linear(inputs))


class GroupedLinear(nn.Module):
    def __init__(self, in_group_size: int, out_group_size: int, groups: int) -> None:
        super().__init__()
        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.groups = groups
        self.weight = nn.Parameter(torch.zeros(groups, in_group_size, out_group_size))
        self.bias = nn.Parameter(torch.zeros(groups, 1, out_group_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        reorganized = (
            inputs.permute(1, 0)
            .reshape(self.groups, self.in_group_size, -1)
            .permute(0, 2, 1)
        )
        hidden = torch.bmm(reorganized, self.weight) + self.bias
        return (
            hidden.permute(0, 2, 1)
            .reshape(self.out_group_size * self.groups, -1)
            .permute(1, 0)
        )


class BranchedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_group_size: int,
        out_group_size: int,
        n_branches: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_branches = n_branches
        self.n_layers = n_layers
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        current_size = in_features
        for index in range(n_layers):
            final = index + 1 == n_layers
            output_size = out_group_size if final else hidden_group_size
            setattr(
                self,
                f"branched_layer_{index + 1}",
                GroupedLinear(current_size, output_size, n_branches),
            )
            current_size = hidden_group_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs.repeat(1, self.n_branches)
        for index in range(self.n_layers - 1):
            hidden = getattr(self, f"branched_layer_{index + 1}")(hidden)
            hidden = self.dropout(self.nonlin(hidden))
        return getattr(self, f"branched_layer_{self.n_layers}")(hidden)


class Malinois(nn.Module):
    """Published BassetBranched architecture for 600-nt reporter-context inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.pad1 = nn.ConstantPad1d((9, 9), 0.0)
        self.conv1 = Conv1dNorm(4, 300, 19)
        self.pad2 = nn.ConstantPad1d((5, 5), 0.0)
        self.conv2 = Conv1dNorm(300, 200, 11)
        self.pad3 = nn.ConstantPad1d((3, 3), 0.0)
        self.conv3 = Conv1dNorm(200, 200, 7)
        self.pad4 = nn.ConstantPad1d((1, 1), 0.0)
        self.maxpool_3 = nn.MaxPool1d(3)
        self.maxpool_4 = nn.MaxPool1d(4)
        self.linear1 = LinearNorm(2600, 1000)
        self.branched = BranchedLinear(1000, 140, 140, 3, 3, 0.5757068086404574)
        self.output = GroupedLinear(140, 1, 3)
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout(0.11625456877954289)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.nonlin(self.conv1(self.pad1(inputs)))
        hidden = self.maxpool_3(hidden)
        hidden = self.nonlin(self.conv2(self.pad2(hidden)))
        hidden = self.maxpool_4(hidden)
        hidden = self.nonlin(self.conv3(self.pad3(hidden)))
        hidden = self.maxpool_4(self.pad4(hidden))
        hidden = torch.flatten(hidden, start_dim=1)
        hidden = self.dropout(self.nonlin(self.linear1(hidden)))
        return self.output(self.branched(hidden))


def load_malinois(
    checkpoint_path: Path, device: str | torch.device = "cpu"
) -> tuple[Malinois, dict[str, object]]:
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = Malinois()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    metadata = {
        "data_module": checkpoint["data_module"],
        "model_module": checkpoint["model_module"],
        "timestamp": checkpoint["timestamp"],
        "random_tag": checkpoint["random_tag"],
        "validation_chromosomes": checkpoint["data_hparams"].val_chrs,
        "test_chromosomes": checkpoint["data_hparams"].test_chrs,
        "input_length": checkpoint["model_hparams"].input_len,
        "inference_device": str(device),
    }
    return model, metadata


@torch.inference_mode()
def predict_sequences(
    model: Malinois,
    sequences: list[str],
    batch_size: int = 64,
    strand_average: bool = True,
) -> np.ndarray:
    predictions = []
    device = next(model.parameters()).device
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        padded = [pad_mpra_sequence(sequence) for sequence in batch]
        forward = model(one_hot(padded).to(device))
        if strand_average:
            reverse = model(
                one_hot([reverse_complement(sequence) for sequence in padded]).to(device)
            )
            forward = (forward + reverse) / 2.0
        predictions.append(forward.cpu().numpy())
    return np.concatenate(predictions, axis=0)


@torch.inference_mode()
def predict_both_strands(
    model: Malinois,
    sequences: list[str],
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return forward and full-reporter reverse-complement predictions."""
    forward_predictions = []
    reverse_predictions = []
    device = next(model.parameters()).device
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        padded = [pad_mpra_sequence(sequence) for sequence in batch]
        forward_predictions.append(model(one_hot(padded).to(device)).cpu().numpy())
        reverse_padded = [reverse_complement(sequence) for sequence in padded]
        reverse_predictions.append(model(one_hot(reverse_padded).to(device)).cpu().numpy())
    return (
        np.concatenate(forward_predictions, axis=0),
        np.concatenate(reverse_predictions, axis=0),
    )
