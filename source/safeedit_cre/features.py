"""Transparent sequence features for non-neural baselines."""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np

from .sequence import normalize_sequence


DNA_BASES = "ACGT"


def kmer_names(k: int) -> tuple[str, ...]:
    if k < 1:
        raise ValueError("k must be positive")
    return tuple("".join(chars) for chars in product(DNA_BASES, repeat=k))


def kmer_frequency_matrix(sequences: Iterable[str], k: int = 4) -> np.ndarray:
    """Return stranded normalized k-mer counts in lexicographic A/C/G/T order."""
    sequences = list(sequences)
    features = np.zeros((len(sequences), 4**k), dtype=np.float32)
    base_code = {base: index for index, base in enumerate(DNA_BASES)}
    high_place = 4 ** (k - 1)

    for row_index, raw_sequence in enumerate(sequences):
        sequence = normalize_sequence(raw_sequence)
        code = 0
        valid_run = 0
        valid_windows = 0
        for base in sequence:
            digit = base_code.get(base)
            if digit is None:
                code = 0
                valid_run = 0
                continue
            if valid_run < k:
                code = code * 4 + digit
                valid_run += 1
                if valid_run < k:
                    continue
            else:
                code = (code % high_place) * 4 + digit
            features[row_index, code] += 1.0
            valid_windows += 1
        if valid_windows:
            features[row_index] /= valid_windows
    return features
