"""DNA sequence normalization and leakage-safe identifiers."""

from __future__ import annotations

import hashlib


DNA_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def normalize_sequence(sequence: str) -> str:
    """Normalize whitespace and case without silently changing bases."""
    return "".join(sequence.split()).upper()


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of an uppercase-normalized DNA string."""
    seq = normalize_sequence(sequence)
    return seq.translate(DNA_COMPLEMENT)[::-1]


def canonical_sequence(sequence: str) -> str:
    """Map a sequence and its reverse complement to one canonical string."""
    seq = normalize_sequence(sequence)
    rc = reverse_complement(seq)
    return min(seq, rc)


def sequence_digest(sequence: str, reverse_complement_aware: bool = False) -> str:
    """Return a stable SHA-256 digest for exact or strand-invariant matching."""
    seq = canonical_sequence(sequence) if reverse_complement_aware else normalize_sequence(sequence)
    return hashlib.sha256(seq.encode("ascii")).hexdigest()


def sequence_qc(sequence: str, expected_length: int | None = 200) -> dict[str, object]:
    """Compute deterministic validation fields for one DNA sequence."""
    seq = normalize_sequence(sequence)
    alphabet = set(seq)
    invalid = sorted(alphabet - set("ACGT"))
    gc = (seq.count("G") + seq.count("C")) / len(seq) if seq else float("nan")
    return {
        "length": len(seq),
        "length_ok": expected_length is None or len(seq) == expected_length,
        "invalid_bases": "".join(invalid),
        "alphabet_ok": not invalid,
        "gc_fraction": gc,
        "exact_sha256": sequence_digest(seq),
        "canonical_sha256": sequence_digest(seq, reverse_complement_aware=True),
    }

