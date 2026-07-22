"""Chromosome-based splits used by the Malinois study."""

from __future__ import annotations


VALIDATION_CHROMOSOMES = {"19", "21", "X"}
TEST_CHROMOSOMES = {"7", "13"}


def normalize_chromosome(chromosome: str) -> str:
    value = str(chromosome).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    return value.upper()


def assign_upstream_split(chromosome: str) -> str:
    chrom = normalize_chromosome(chromosome)
    if chrom == "SYNTH":
        return "synthetic_excluded"
    if chrom in VALIDATION_CHROMOSOMES:
        return "validation"
    if chrom in TEST_CHROMOSOMES:
        return "test"
    return "train"

